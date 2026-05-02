"""
MoE AGENT — Mixture of Experts (Agent 4 of 8)

Pipeline:
Input → Router Mechanism → [Expert 1 ‖ Expert 2 ‖ Expert 3 ‖ Expert 4] → Top-k Selection → Advanced Patterning → Quantization → Output

Standard: Production-Grade | Pydantic v2 | ABC | Retry Logic

Core Philosophy:
Sparse activation — only Top-k of N experts fire per query.
Each expert is a fully specialised reasoning persona.
The router learns to dispatch; the synthesiser fuses outputs.
"""

import os, time, uuid, json, re, numpy as np
from openai import OpenAI, RateLimitError, APITimeoutError, APIError, APIConnectionError
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
load_dotenv()

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="moe.log")

# ==========================================
# Variable Configuration
# ==========================================
from utils.state_checkpointer import StateCheckpointer
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
checkpointer = StateCheckpointer(
    directory=CHECKPOINT_DIR, 
    filename="moe_checkpoint.json",
    logger=logger
)

TOKEN = os.environ['GITHUB_TOKEN']
ENDPOINT = os.environ['GITHUB_ENDPOINT']
CHAT_MODEL = os.environ['GITHUB_MODEL_NAME']
EMBEDDING_MODEL = os.environ['GITHUB_EMBED_NAME']

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

NUM_EXPERTS = 4  # total experts in the pool
TOP_K = 2  # sparse activation: fire only top-k experts
CONFIDENCE_THRESHOLD = 0.15  # minimum router score for expert eligibility
QUANTIZATION_BITS = 8  # scalar quantization of gating weights
CODEBOOK_SIZE = 256  # 2^8

# Thread pool for parallel expert inference
MAX_EXPERT_WORKERS   = 4

# prompt templates
ROUTER_PROMPT = """Query: {moe_input}

Available Experts:
{experts_desc}

For each expert, assign a relevance score and brief rationale.

Respond with:
{{
  "scores": {{
    "E1": {{"raw_score": <0.0-1.0>, "rationale": "<why>"}},
    "E2": {{"raw_score": <0.0-1.0>, "rationale": "<why>"}},
    "E3": {{"raw_score": <0.0-1.0>, "rationale": "<why>"}},
    "E4": {{"raw_score": <0.0-1.0>, "rationale": "<why>"}}
  }}
}}"""

ADVANCED_PATTERNING_PROMPT = """Original Query: {query}

Expert Outputs:
{experts_block}

Identify cross-expert patterns and extract consensus/divergence points.

Respond with:
{{
  "patterns": [
    {{
      "pattern_id": "PAT001",
      "experts_involved": ["E1", "E2"],
      "pattern_type": "<convergence|divergence|complementary|contradiction>",
      "description": "<what the pattern reveals>",
      "significance": <0.0-1.0>
    }}
  ],
  "consensus_points": ["<point 1>", "<point 2>"],
  "divergence_points": ["<point 1>"]
}}"""

# ==========================================
# ENUMS
# ==========================================
class PipelineStage(str, Enum):
    INPUT = "INPUT"
    ROUTER_MECHANISM = "ROUTER_MECHANISM"
    EXPERT_INFERENCE = "EXPERT_INFERENCE"
    TOP_K_SELECTION = "TOP_K_SELECTION"
    ADVANCED_PATTERNING = "ADVANCED_PATTERNING"
    QUANTIZATION = "QUANTIZATION"
    OUTPUT = "OUTPUT"


class ExpertDomain(str, Enum):
    REASONING = "REASONING"
    DOMAIN_KNOWLEDGE = "DOMAIN_KNOWLEDGE"
    CREATIVE_SYNTHESIS = "CREATIVE_SYNTHESIS"
    CRITICAL_ANALYSIS = "CRITICAL_ANALYSIS"


class ExpertStatus(str, Enum):
    STANDBY = "STANDBY"
    ACTIVATED = "ACTIVATED"
    SKIPPED = "SKIPPED"
    FAILED = "FAILED"


class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"

# ==========================================
# EXPERT REGISTRY  —  Static configuration for each expert
# ==========================================
EXPERT_REGISTRY: Dict[str, dict] = {
    "E1": {
        "id": "E1",
        "name": "Expert 1 — Reasoning & Logic",
        "domain": ExpertDomain.REASONING,
        "emoji": "🧮",
        "system": (
            "You are Expert 1: a rigorous reasoning and formal logic specialist. "
            "Your approach: decompose problems into first principles, apply deductive and inductive reasoning, identify logical fallacies, and construct step-by-step arguments. "
            "Always show your reasoning chain explicitly. "
            "Be precise, structured, and conclusive."
        ),
        "temperature": 0.2,
        "max_tokens" : 600,
        "strengths": ["logic", "mathematics", "formal reasoning",
                    "step-by-step analysis", "deduction", "proofs"],
    },
    "E2": {
        "id": "E2",
        "name": "Expert 2 — Domain Knowledge",
        "domain": ExpertDomain.DOMAIN_KNOWLEDGE,
        "emoji": "📚",
        "system": (
            "You are Expert 2: a vast domain knowledge encyclopaedia spanning science, technology, history, medicine, law, and economics. "
            "Your approach: cite relevant facts, contextualise within domain frameworks, surface edge cases, and provide authoritative depth. "
            "Prioritise accuracy and comprehensiveness over brevity."
        ),
        "temperature": 0.1,
        "max_tokens": 700,
        "strengths": ["factual recall", "domain expertise", "context",
                    "definitions", "history", "scientific knowledge"],
    },
    "E3": {
        "id": "E3",
        "name": "Expert 3 — Creative Synthesis",
        "domain": ExpertDomain.CREATIVE_SYNTHESIS,
        "emoji": "🎨",
        "system": (
            "You are Expert 3: a lateral thinking and creative synthesis engine. "
            "Your approach: draw unexpected connections across disciplines, generate novel framings, use analogies and metaphors, propose unconventional solutions, and synthesise ideas from diverse fields. "
            "Think divergently before converging."
        ),
        "temperature": 0.85,
        "max_tokens": 600,
        "strengths": ["creativity", "analogies", "cross-domain thinking",
                    "novel ideas", "metaphors", "brainstorming"],
    },
    "E4": {
        "id": "E4",
        "name": "Expert 4 — Critical Analysis",
        "domain": ExpertDomain.CRITICAL_ANALYSIS,
        "emoji": "🔎",
        "system": (
            "You are Expert 4: a rigorous critical analysis and adversarial evaluation specialist. "
            "Your approach: identify assumptions, probe weaknesses, steelman opposing views, surface risks and limitations, and stress-test conclusions. "
            "Be constructively adversarial — our goal is to strengthen understanding by finding what's missing."
        ),
        "temperature": 0.3,
        "max_tokens": 600,
        "strengths": ["critique", "risk analysis", "assumptions",
                    "counterarguments", "limitations", "stress-testing"],
    },
}

# ==========================================
# PYDANTIC MODELS
# ==========================================
# stage 1: input
class MoEInput(BaseModel):
    """Stage 1 — Validated raw input to the MoE pipeline."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the request.")
    query: str = Field(..., min_length=3, max_length=16000 , description="The user's query or problem statement.")
    top_k: int = Field(default=TOP_K, ge=1, le=NUM_EXPERTS, description="Number of top experts to activate (sparse k).")
    require_experts: List[str] = Field(default_factory=list, description="Force-activate specific expert IDs (e.g. ['E1', 'E3']).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual metadata for routing decisions.")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("❌ Query cannot be empty or whitespace.")
        return stripped
    
    @field_validator("require_experts")
    @classmethod
    def validate_required_experts(cls, v: List[str]) -> List[str]:
        valid_ids = set(EXPERT_REGISTRY.keys())
        for expert_id in v:
            if expert_id not in valid_ids:
                raise ValueError(f"❌ Invalid expert ID in require_experts: {expert_id}. Valid IDs: {sorted(valid_ids)}")
        return v
    
    @model_validator(mode="after")
    def stamp_metadata(self) -> "MoEInput":
        self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        return self
    
# staege 2: router mechanism
class ExpertScore(BaseModel):
    """Represents the router's confidence score for a specific expert."""
    expert_id: str = Field(..., description="ID of the expert (e.g. 'E1').")
    expert_name: str = Field(..., description="Name of the expert (e.g. 'Expert 1 — Reasoning & Logic').")
    domain: ExpertDomain = Field(..., description="Domain of expertise (e.g. REASONING, DOMAIN_KNOWLEDGE).")
    raw_score: float = Field(..., ge=0.0, le=1.0, description="Raw relevance score before softmax (0.0 to 1.0).")
    softmax_score: float = Field(..., ge=0.0, le=1.0, description="Softmax-normalized score representing routing confidence.")
    rationale: str = Field(..., description="Explanation of why this expert is relevant to the query.")
    is_eligible: bool = Field(..., description="True if softmax_score >= CONFIDENCE_THRESHOLD={CONFIDENCE_THRESHOLD}, else False.")

class RouterResult(BaseModel):
    """Stage 2 — Router Mechanism output."""
    stage: PipelineStage = Field(default=PipelineStage.ROUTER_MECHANISM, description="Pipeline stage identifier.")
    expert_scores: List[ExpertScore] = Field(..., description="List of router scores for each expert.")
    top_k_ids: List[str] = Field(..., description="IDs of the top-k experts selected for activation.")
    router_entropy: float = Field(..., ge=0.0, description="Shannon entropy of softmax distribution of the router's score distribution, indicating confidence spread.")
    load_balance: float = Field(..., ge=0.0, le=1.0, description="Measure of load balance across experts, calculated as 1 - (max softmax_score - min softmax_score). Closer to 1 indicates more balanced routing.")
    processing_time: float = Field(..., ge=0.0, description="Time taken to execute the routing mechanism in seconds.")

# stage 3: expert inference
class ExpertOutput(BaseModel):
    """Represents the output from an individual expert after processing the query."""
    expert_id: str = Field(..., description="ID of the expert (e.g. 'E1').")
    expert_name: str = Field(..., description="Name of the expert (e.g. 'Expert 1 — Reasoning & Logic').")
    domain: ExpertDomain = Field(..., description="Domain of expertise (e.g. REASONING, DOMAIN_KNOWLEDGE).")
    emoji: str = Field(..., description="Emoji representing the expert's domain (e.g. '🧮').")
    status: ExpertStatus = Field(..., description="Processing status of the expert (STANDBY, ACTIVATED, SKIPPED, FAILED).")
    router_score: float = Field(..., ge=0.0, le=1.0, description="Softmax score assigned by the router, indicating relevance to the query.")
    response: str = Field(..., description="The raw text response generated by the expert.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Expert's self-assessed confidence in its response (0.0 to 1.0).")
    key_insights: List[str] = Field(default_factory=list, description="List of key insights or takeaways extracted from the expert's response.")
    processing_time: float = Field(..., ge=0.0, description="Time taken for the expert to process and generate a response in seconds.")

class ExpertInferenceResult(BaseModel):
    """Stage 3 — All expert outputs (activated + skipped)."""
    stage: PipelineStage = Field(default=PipelineStage.EXPERT_INFERENCE, description="Pipeline stage identifier.")
    all_outputs: List[ExpertOutput] = Field(..., description="List of outputs from all experts, including those that were activated and those that were skipped.")
    activated_count: int = Field(..., description="Number of experts that were activated and generated responses.")
    skipped_count: int = Field(..., description="Number of experts that were skipped based on router scores or require_experts criteria.")
    processing_time: float = Field(..., ge=0.0, description="Total time taken for the expert inference stage in seconds.")

# stage 4: top-k selection
class SelectedExpert(BaseModel):
    """Represents an expert selected in the top-k selection stage with normalised weight."""
    rank: int = Field(..., ge=1, description="Rank of the expert among the top-k selected (1 for highest score).")
    expert_id: str = Field(..., description="ID of the expert (e.g. 'E1').")
    expert_name: str = Field(..., description="Name of the expert (e.g. 'Expert 1 — Reasoning & Logic').")
    domain: ExpertDomain = Field(..., description="Domain of expertise (e.g. REASONING, DOMAIN_KNOWLEDGE).")
    router_weight: float = Field(..., ge=0.0, le=1.0, description="Softmax score assigned by the router, indicating relevance to the query (normalised weight among top-k).")
    response: str = Field(..., description="The raw text response generated by the expert.")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Expert's self-assessed confidence in its response (0.0 to 1.0).")
    key_insights: List[str] = Field(default_factory=list, description="List of key insights or takeaways extracted from the expert's response.")

class TopKSelectionResult(BaseModel):
    """Stage 4 — Final top-k selected experts with normalised weights."""
    stage: PipelineStage = Field(default=PipelineStage.TOP_K_SELECTION, description="Pipeline stage identifier.")
    k: int = Field(..., description="Number of top experts selected (k).")
    selected_experts: List[SelectedExpert] = Field(..., description="List of the top-k selected experts with their responses and normalised router weights.")
    weight_sum: float = Field(..., ge=0.0, description="Sum of the router weights of the selected experts, used for normalisation (should be ≈ 1.0 after normalisation).")
    selection_entropy: float = Field(..., ge=0.0, description="Shannon entropy of the normalised router weights among the selected experts, indicating confidence spread within the top-k selection (entropy of top-k weight distribution).")
    processing_time: float = Field(..., ge=0.0, description="Time taken for the top-k selection stage in seconds.")

# stage 5: advanced patterning
class CrossExpertPattern(BaseModel):
    """Represents a detected pattern or connection across multiple expert responses."""
    pattern_id: str = Field(..., description="Unique identifier for the detected pattern.")
    experts_involved: List[str] = Field(..., description="List of expert IDs whose responses are involved in this pattern.")
    pattern_type: str = Field(..., description="Type of pattern detected (e.g. 'convergence', 'divergence', 'complementary', 'contradiction').")
    description: str = Field(..., description="Natural language description of the detected pattern and its significance.")
    significance: float = Field(..., ge=0.0, le=1.0, description="Quantitative measure of the pattern's significance or strength (0.0 to 1.0).")

class AdvancedPatterningResult(BaseModel):
    """Stage 5 — Detected cross-expert patterns and connections."""
    stage: PipelineStage = Field(default=PipelineStage.ADVANCED_PATTERNING, description="Pipeline stage identifier.")
    patterns: List[CrossExpertPattern] = Field(..., description="List of detected patterns or connections across expert responses.")
    consensus_points: List[str] = Field(default_factory=list, description="List of key points where multiple experts converge or agree.")
    divergence_points: List[str] = Field(default_factory=list, description="List of key points where experts diverge or contradict each other.")
    synthesised_answer: str = Field(..., description="A synthesized answer or insight generated by integrating the top-k expert responses, informed by the detected patterns.")
    processing_time: float = Field(..., ge=0.0, description="Time taken for the advanced patterning stage in seconds.")

# stage 6: quantization
class QuantizationResult(BaseModel):
    """Stage 6 — Quantization of expert responses and router weights."""
    stage: PipelineStage = Field(default=PipelineStage.QUANTIZATION, description="Pipeline stage identifier.")
    bits: int = Field(..., description="Number of bits used for scalar quantization (e.g. 8).")
    codebook_size: int = Field(..., description="Size of the quantization codebook (e.g. 256 for 8 bits).")
    original_weights: List[float] = Field(..., description="List of original router weights for the top-k selected experts before quantization.")
    quantized_codes: List[int] = Field(..., description="List of quantized codes (integers) corresponding to the original weights after scalar quantization.")
    reconstruction_error: float = Field(..., ge=0.0, description="Mean squared error between original weights and reconstructed weights from quantized codes, indicating quantization fidelity.")   
    compression_ratio: float = Field(..., ge=0.0, description="Ratio of the size of original weights to the size of quantized representation, indicating compression efficiency.")
    processing_time: float = Field(..., ge=0.0, description="Time taken for the quantization stage in seconds.")

# stage 7: output
class MoEOutput(BaseModel):
    """Stage 7 — Final output of the MoE pipeline."""
    stage: PipelineStage = Field(default=PipelineStage.OUTPUT, description="Pipeline stage identifier.")
    request_id: str = Field(..., description="Unique ID for the request, matching the input request_id for traceability.")
    status: ProcessingStatus = Field(..., description="Overall processing status of the MoE pipeline (PENDING, SUCCESS, FAILED).")
    
    # stage payloads
    router: RouterResult = Field(..., description="Output from the router mechanism stage.")
    expert_inference: ExpertInferenceResult = Field(..., description="Output from the expert inference stage.")
    top_k_selection: TopKSelectionResult = Field(..., description="Output from the top-k selection stage.")
    advanced_patterning: AdvancedPatterningResult = Field(..., description="Output from the advanced patterning stage.")
    quantization: QuantizationResult = Field(..., description="Output from the quantization stage.")

    # final synthesized answer
    final_answer: str = Field(..., description="The final synthesized answer or insight generated by the MoE pipeline after integrating expert responses and detected patterns.")
    expert_attribution: List[str] = Field(default_factory=list, description="List of expert IDs that contributed to the final answer, for attribution purposes.")
    total_pipeline_time: float = Field(..., ge=0.0, description="Total time taken for the entire MoE pipeline from input to final output in seconds.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the processing, such as timestamps, model versions, or debug info.")

    @model_validator(mode="after")
    def populate_metadata(self) -> "MoEOutput":
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["experts_activated"] = self.expert_inference.activated_count
        self.metadata["top_k"] = self.top_k_selection.k
        self.metadata["router_entropy"] = self.router.router_entropy
        return self

# ==========================================
# ABSTRACT BASE  (shared across all 8 agents)
# ==========================================
class BaseAIAgent(ABC):
    """Abstract base class defining the shared contract for all 8 AI agents."""

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        if client is not None:
            self.client = client
        else:
            self.client = OpenAI(base_url=ENDPOINT, api_key=TOKEN)
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Core execution pipeline to be implemented by each specialized agent."""
        ...

    def _retry_api_call(self, fn: Callable, *args, **kwargs) -> Any:
        """
        Exponential-backoff retry wrapper for any OpenAI API call.
        Handles: RateLimitError, APITimeoutError, APIError.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except RateLimitError as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning(
                    f"🚨 Rate-limit (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait:.1f}s…",
                    extra={"tag": "retry"}
                )
                time.sleep(wait)
                last_exc = e
            except APIConnectionError as e:       # FIX-05
                wait = RETRY_BACKOFF_BASE * attempt
                logger.warning(
                    f"🚨 Connection error (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait:.1f}s…",
                    extra={"tag": "retry"}
                )
                time.sleep(wait)
                last_exc = e
            except APITimeoutError as e:
                wait = RETRY_BACKOFF_BASE * attempt
                logger.warning(
                    f"🚨 Timeout (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait:.1f}s…",
                    extra={"tag": "retry"}
                )
                time.sleep(wait)
                last_exc = e
            except APIError as e:
                logger.error(f"🚨 APIError on attempt {attempt}: {e}", extra={"tag": "fail"})
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_BACKOFF_BASE ** attempt)
                last_exc = e
        raise RuntimeError(f"🚨 All {MAX_RETRIES} API retry attempts exhausted.") from last_exc
    
    def _gpt_json_response(self, system: str, user: str, max_tokens: int = 1500, temperature: float = 0.2) -> dict:
        """wrapper for GPT call with JSON response format."""
        response = self._retry_api_call(
            self.client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"}
        )
        raw = (response.choices[0].message.content or "{}").strip()
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        
        try:
            data = json.loads(clean)
            logger.debug(f"🔍 parsed json: {data}")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"JSONDecodeError: {e} | Attempting extraction. Raw: {raw}")
            # Attempt to extract innermost or bounds of { }
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(clean[start:end+1])
                    logger.debug(f"🔍 recovered json: {data}")
                    return data
                except json.JSONDecodeError:
                    pass
                    
            logger.error("Failed to recover JSON, returning empty dict to prevent crash.")
            return {}

    def _gpt_text_response(self, system: str, user: str, max_tokens: int = 800, temperature: float = 0.3) -> str:
        """wrapper for GPT call with plain text response format."""
        response = self._retry_api_call(
            self.client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=temperature
        )
        res = (response.choices[0].message.content or "").strip()
        logger.debug(f"🔍 raw gpt text response: {res}")
        return res
    
# ==========================================
# HELPER — Numerics
# ==========================================
def softmax(scores: list[float]) -> list[float]:
    """Numerically stable softmax."""
    arr = np.array(scores, dtype=np.float64)
    shifted = arr - arr.max()
    exp_arr = np.exp(shifted)
    return (exp_arr / exp_arr.sum()).tolist()

def shannon_entropy(probs: list[float]) -> float:
    """Shannon entropy H(p) = -Σ p·log(p)."""
    arr = np.array(probs, dtype=np.float64)
    arr = arr[arr > 0]
    return float(-np.sum(arr * np.log2(arr)))

def scalar_quantize(
    weights: list[float],
    bits: int = QUANTIZATION_BITS,
) -> tuple[list[int], float, float]:
    """
    Uniform scalar quantization.
    Returns: (codes, reconstruction_error, compression_ratio)
    """
    arr = np.array(weights, dtype=np.float64)
    levels = 2 ** bits
    v_min, v_max = arr.min(), arr.max()
    v_range = v_max - v_min + 1e-10
    codes = np.floor((arr - v_min) / v_range * (levels - 1)) \
                .astype(np.int32).clip(0, levels - 1)
    # Dequantize to measure reconstruction error
    dequantized = codes.astype(np.float64) / (levels - 1) * v_range + v_min
    recon_error = float(np.mean((arr - dequantized) ** 2))
    original_bytes = len(arr) * 8       # float64 = 8 bytes
    compressed_bytes = len(codes) * 1   # int8   = 1 byte
    compression_ratio = original_bytes / compressed_bytes
    return codes.tolist(), recon_error, compression_ratio

# ==========================================
# PIPELINE STAGESs
# ==========================================
class RouterMechanismStage:
    """
    Stage 2: ROUTER MECHANISM
    The gating network — scores each expert's relevance to the query,
    applies softmax normalisation, and selects the top-k eligible experts.

    Architecture fidelity:
    - True MoE routers are learned linear layers trained end-to-end.
    - We replicate this with a GPT-4o scoring pass that produces
      calibrated relevance scores, then apply softmax + top-k selection.
    - Router entropy measures dispatch diversity (high = balanced load).
    - Load balance (std-dev) measures fairness across experts.
    """
    def run(self, moe_input: MoEInput, agent: BaseAIAgent) -> RouterResult:
        logger.info(
            f"⚙️ [ROUTER] Scoring {NUM_EXPERTS} experts for query routing..."
        )
        t0 = time.perf_counter()

        experts_desc = "\n".join(
            f"  {eid}: {edef['name']} | "
            f"Strengths: {', '.join(edef['strengths'][:4])}"
            for eid, edef in EXPERT_REGISTRY.items()
        )

        data = agent._gpt_json_response(
            system=(
                "You are a neural router for a Mixture of Experts system. "
                "Evaluate how relevant each expert is to the given query. "
                "Output calibrated float scores [0.0-1.0]. JSON only."
            ),
            user=ROUTER_PROMPT.format(
                moe_input=moe_input.query,
                experts_desc=experts_desc,
            ),
            max_tokens=800,
            temperature=0.1
        )

        raw_scores_map = data.get("scores", {})

        # extract raw scores
        raw_scores = {
            eid: float(raw_scores_map.get(eid, {}).get("raw_score", 0.25))
            for eid in EXPERT_REGISTRY
        }

        # apply softmax
        score_list = [raw_scores[eid] for eid in EXPERT_REGISTRY]
        softmax_list = softmax(score_list)
        softmax_map = dict(zip(EXPERT_REGISTRY.keys(), softmax_list))

        # build expert scores onjects
        expert_scores: List[ExpertScore] = []
        for eid, edef in EXPERT_REGISTRY.items():
            sm_score = softmax_map[eid]
            expert_scores.append(ExpertScore(
                expert_id=eid,
                expert_name=edef["name"],
                domain=edef["domain"],
                raw_score=raw_scores[eid],
                softmax_score=sm_score,
                rationale=raw_scores_map.get(eid, {}).get("rationale", ""),
                is_eligible=sm_score >= CONFIDENCE_THRESHOLD,
            ))
        
        # top-k selection
        sorted_scores = sorted(
            expert_scores, key=lambda x: x.softmax_score, reverse=True
        )
        top_k_ids: list[str] = []

        # enforce require_experts first
        for req_id in moe_input.require_experts:
            if req_id not in top_k_ids:
                top_k_ids.append(req_id)

        # fill remaining slots with highest softmax scores
        for es in sorted_scores:
            if len(top_k_ids) >= moe_input.top_k:
                break
            if es.expert_id not in top_k_ids and es.is_eligible:
                top_k_ids.append(es.expert_id)

        # fallback: if not enough eligible experts, fill with highest raw scores
        if len(top_k_ids) < moe_input.top_k:
            for es in sorted_scores:
                if es.expert_id not in top_k_ids:
                    top_k_ids.append(es.expert_id)
                if len(top_k_ids) >= moe_input.top_k:
                    break

        router_entropy = shannon_entropy(softmax_list)
        load_balance = float(np.std(softmax_list))
        elapsed = time.perf_counter() - t0

        result = RouterResult(
            expert_scores=expert_scores,
            top_k_ids=top_k_ids,
            router_entropy=round(router_entropy, 4),
            load_balance=round(load_balance, 4),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [ROUTER] top_k={top_k_ids} | entropy={router_entropy:.4f} | "
            f"load_balance={load_balance:.4f} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [ROUTER] expert_scores...")
        for es in expert_scores:
            icon = "🟢" if es.expert_id in top_k_ids else "⚪"
            logger.debug(
                f"🔍 {icon} {es.expert_id}: raw={es.raw_score:.3f} "
                f"softmax={es.softmax_score:.3f} | {es.rationale[:50]}"
            )
        return result
    
class ExpertInferenceStage:
    """
    Stage 3: EXPERT INFERENCE
    Runs activated experts in parallel (ThreadPoolExecutor).
    Non-activated experts are marked SKIPPED — sparse activation preserved.

    Each expert has:
    - A distinct system prompt (persona)
    - Calibrated temperature (logic=0.2, knowledge=0.1, creative=0.85, critical=0.3)
    - Self-assessed confidence score
    - Extracted key insights
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def _run_single_expert(self, expert_id: str, query: str, router_score: float, retry_fn) -> ExpertOutput:
        edef = EXPERT_REGISTRY[expert_id]
        logger.info(
            f"⚙️ [{edef['emoji']} EXPERT {expert_id}] "
            f"Activating {edef['name']} (score={router_score:.3f})..."
        )
        t0 = time.perf_counter()

        try:
            response = retry_fn(
                self._client.chat.completions.create,
                model=CHAT_MODEL,
                messages   = [
                    {"role": "system", "content": edef["system"]},
                    {
                        "role": "user",
                        "content": (
                            f"Query: {query}\n\n"
                            "Provide your expert response. At the end, on separate lines add:\n"
                            "CONFIDENCE: <float 0.0-1.0>\n"
                            "INSIGHTS: <bullet 1> | <bullet 2> | <bullet 3>"
                        ),
                    },
                ],
                max_tokens=edef["max_tokens"],
                temperature=edef["temperature"]
            )

            raw_text = response.choices[0].message.content or ""
            confidence = 0.8
            key_insights: List[str] = []

            # parse confidence and insights from the end of the response
            lines = raw_text.strip().splitlines()
            clean_lines = []
            for line in lines:
                if line.startswith("CONFIDENCE:"):
                    try:
                        confidence = float(line.split(":")[1].strip())
                    except (ValueError, IndexError):
                        pass
                elif line.startswith("INSIGHTS:"):
                    insights_raw = line.replace("INSIGHTS:", "").strip()
                    key_insights = [i.strip() for i in insights_raw.split("|") if i.strip()]
                else:
                    clean_lines.append(line)

            response_text = "\n".join(clean_lines).strip()
            elapsed = time.perf_counter() - t0

            logger.info(
                f"✅ [{edef['emoji']} EXPERT {expert_id}] Complete | confidence={min(max(confidence, 0.0), 1.0)} | "
                f"insights={len(key_insights)} | time={elapsed:.4f}s"
            )
            logger.debug(
                f"🔍 [{edef['emoji']} EXPERT {expert_id}] | response_text: {response_text} | "
                f"key_insights: {key_insights} | router_score={router_score:.3f}"
            )

            return ExpertOutput(
                expert_id=expert_id,
                expert_name=edef["name"],
                domain=edef["domain"],
                emoji=edef["emoji"],
                status=ExpertStatus.ACTIVATED,
                router_score=router_score,
                response=response_text,
                confidence=min(max(confidence, 0.0), 1.0),
                key_insights=key_insights,
                processing_time=round(elapsed, 4),
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            logger.error(
                f"❌ [{edef['emoji']} EXPERT {expert_id}] FAILED: {e}"
            )
            return ExpertOutput(
                expert_id=expert_id,
                expert_name=edef["name"],
                domain=edef["domain"],
                emoji=edef["emoji"],
                status=ExpertStatus.FAILED,
                router_score=router_score,
                processing_time= round(elapsed, 4),
            )

    def run(self, router: RouterResult, query: str, retry_fn) -> ExpertInferenceResult:
        logger.info(
            f"⚙️ [EXPERT INFERENCE] Activating {len(router.top_k_ids)}/{NUM_EXPERTS} | "
            f"experts in parallel: {router.top_k_ids}"
        )
        t0 = time.perf_counter()

        score_map = {es.expert_id: es.softmax_score for es in router.expert_scores}
        activated_ids = set(router.top_k_ids)
        all_outputs: List[ExpertOutput] = []

        # parallel execution of activated experts
        with ThreadPoolExecutor(max_workers=MAX_EXPERT_WORKERS) as executor:
            futures = {
                executor.submit(
                    self._run_single_expert,
                    eid,
                    query,
                    score_map.get(eid, 0.0),
                    retry_fn,
                ): eid
                for eid in activated_ids
            }
            active_results: dict[str, ExpertOutput] = {}
            for future in as_completed(futures):
                eid = futures[future]
                result = future.result()
                active_results[eid] = result
            
        # build final outputs list, preserving order and marking skipped experts
        for eid in EXPERT_REGISTRY:
            if eid in active_results:
                all_outputs.append(active_results[eid])
            else:
                edef = EXPERT_REGISTRY[eid]
                logger.debug(f"🔍 ⚪ [EXPERT {eid}] SKIPPED (not in top-k)")
                all_outputs.append(ExpertOutput(
                    expert_id=eid,
                    expert_name=edef["name"],
                    domain=edef["domain"],
                    emoji=edef["emoji"],
                    status=ExpertStatus.SKIPPED,
                    router_score=score_map.get(eid, 0.0),
                    response="SKIPEPD",
                    confidence=0.0,
                    key_insights=[],
                    processing_time=0.0,
                ))

        activated_count = sum(1 for o in all_outputs if o.status == ExpertStatus.ACTIVATED)
        skipped_count = sum(1 for o in all_outputs if o.status == ExpertStatus.SKIPPED)
        total_time = time.perf_counter() - t0

        result = ExpertInferenceResult(
            all_outputs=all_outputs,
            activated_count=activated_count,
            skipped_count=skipped_count,
            processing_time=round(total_time, 4)
        )
        logger.info(
            f"✅ [EXPERT INFERENCE] activated={activated_count} | "
            f"skipped={skipped_count} | parallel_time={total_time:.4f}s"
        )
        logger.debug(
            f"🔍 [EXPERT INFERENCE] | number_of_outputs={len(all_outputs)} | all_output={all_outputs[:5]})"
        )
        return result
    
class TopKSelectionStage:
    """
    Stage 4: TOP-K SELECTION
    Filters expert outputs to the top-k activated experts,
    normalises their router weights to sum to 1.0,
    and ranks them by weighted confidence.

    Sparsity invariant: exactly k experts contribute to downstream stages.
    """
    def run(self, expert_inference: ExpertInferenceResult, k: int) -> TopKSelectionResult:
        logger.info(f"⚙️  [TOP-K SELECTION] Selecting top-{k} experts...")
        t0 = time.perf_counter()

        activated = [
            o for o in expert_inference.all_outputs
            if o.status == ExpertStatus.ACTIVATED
        ]

        # rank by router score × confidence (combined quality signal)
        ranked = sorted(
            activated,
            key=lambda o: o.router_score * o.confidence,
            reverse=True,
        )[:k]

        # normalise weights to sum = 1.0
        weight_total = sum(o.router_score for o in ranked) or 1.0
        selected: list[SelectedExpert] = []
        for rank_idx, expert_out in enumerate(ranked, start=1):
            selected.append(SelectedExpert(
                rank=rank_idx,
                expert_id=expert_out.expert_id,
                expert_name=expert_out.expert_name,
                domain=expert_out.domain,
                router_weight=round(expert_out.router_score / weight_total, 4),
                response=expert_out.response,
                confidence=expert_out.confidence,
                key_insights=expert_out.key_insights,
            ))

        weight_sum       = round(sum(s.router_weight for s in selected), 4)
        weights          = [s.router_weight for s in selected]
        selection_entropy = shannon_entropy(weights)
        elapsed           = time.perf_counter() - t0

        result = TopKSelectionResult(
            k=k,
            selected_experts=selected,
            weight_sum=weight_sum,
            selection_entropy=round(selection_entropy, 4),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [TOP-K SELECTION] k={len(selected)} | weight_sum={weight_sum} | "
            f"entropy={selection_entropy:.4f} | time={elapsed:.4f}s"
        )
        for s in selected:
            logger.debug(
                f"🔍 Rank {s.rank}: {s.expert_id} — {s.expert_name} "
                f"(weight={s.router_weight:.4f}, conf={s.confidence:.2f})"
            )
        return result

class AdvancedPatterningStage:
    """
    Stage 5: ADVANCED PATTERNING
    Meta-synthesis over top-k expert outputs.

    Identifies:
    - Convergence: where experts independently reached the same conclusion
    - Divergence: where experts disagreed or emphasised different aspects
    - Complementary: where expert outputs are non-overlapping but compatible
    - Contradiction: genuine logical conflicts between expert outputs

    Then synthesises a final fused answer weighted by router scores.
    """
    def run(self, top_k: TopKSelectionResult, query: str, agent: BaseAIAgent) -> AdvancedPatterningResult:
        logger.info(
            f"⚙️ [ADVANCED PATTERNING] Analysing patterns across {top_k.k} expert outputs..."
        )
        t0 = time.perf_counter()

        experts_block = "\n\n".join([
            f"--- {s.expert_name} (weight={s.router_weight:.3f}, "
            f"confidence={s.confidence:.2f}) ---\n{s.response}"
            for s in top_k.selected_experts
        ]) 

        pattern_data = agent._gpt_json_response(
            system=(
                "You are a meta-synthesis engine for a Mixture of Experts system. "
                "Analyse multiple expert outputs and identify cross-expert patterns. "
                "Output a JSON object only."
            ),
            user=ADVANCED_PATTERNING_PROMPT.format(
                query=query,
                experts_block=experts_block,
            ),
            max_tokens=1500,
            temperature=0.2
        )
        patterns = [
            CrossExpertPattern(**p)
            for p in pattern_data.get("patterns", [])
        ]
        consensus_points = pattern_data.get("consensus_points", [])
        divergence_points = pattern_data.get("divergence_points", [])

        # weighted synthesis of expert outputs
        weights_text = "\n".join(
            f"Expert {s.expert_id} ({s.expert_name}) "
            f"contributes with weight {s.router_weight:.3f}:\n  {s.response[:300]}"
            for s in top_k.selected_experts
        )
        insights_text = "\n".join(
            f"[{s.expert_id}] " + " | ".join(s.key_insights[:3])
            for s in top_k.selected_experts
        )
        synthesised_answer = agent._gpt_text_response(
            system=(
                "You are the final synthesis layer of a Mixture of Experts system. "
                "Produce a single, coherent, comprehensive answer by fusing all expert perspectives according to their weights. "
                "Integrate all insights; do not merely summarise. Write in a clear, authoritative tone."
            ),
            user=(
                f"Query: {query}\n\n"
                f"Expert Contributions (weighted):\n{weights_text}\n\n"
                f"Key Insights:\n{insights_text}\n\n"
                f"Consensus Points: {', '.join(consensus_points)}\n"
                f"Divergence Points: {', '.join(divergence_points)}\n\n"
                "Synthesise a complete, expert-fused answer:"
            ),
            max_tokens=1_000,
            temperature=0.4,
        )

        elapsed = time.perf_counter() - t0
        result = AdvancedPatterningResult(
            patterns=patterns,
            consensus_points=consensus_points,
            divergence_points=divergence_points,
            synthesised_answer=synthesised_answer,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [ADVANCED PATTERNING] patterns={len(patterns)} | consensus={len(consensus_points)} | "
            f"divergence={len(divergence_points)} | time={elapsed:.4f}s"
        )
        logger.debug(
            f"🔍 [ADVANCED PATTERNING] synthesised_answer: {synthesised_answer[:300]} | patterns={patterns[:5]}"
        )
        return result

class QuantizationStage:
    """
    Stage 6: QUANTIZATION
    Quantizes the router's full softmax weight distribution (all N experts)
    into discrete scalar codes.

    In production MoE systems, quantization is applied to expert weights
    to reduce memory footprint during inference. We replicate this faithfully
    on the gating probabilities.
    """
    def run(self, router: RouterResult) -> QuantizationResult:
        logger.info(
            f"⚙️ [QUANTIZATION] Quantizing router weights ({QUANTIZATION_BITS}-bit)..."
        )
        t0 = time.perf_counter()

        original_weights = [es.softmax_score for es in router.expert_scores]
        codes, recon_error, compression_ratio = scalar_quantize(
            original_weights, bits=QUANTIZATION_BITS
        )
        elapsed = time.perf_counter() - t0
        result = QuantizationResult(
            bits=QUANTIZATION_BITS,
            codebook_size=CODEBOOK_SIZE,
            original_weights=original_weights,
            quantized_codes=codes,
            reconstruction_error=round(recon_error, 6),
            compression_ratio=round(compression_ratio, 2),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [QUANTIZATION] codes={codes} | reconstruction_error={recon_error:.8f} | "
            f"compression_ratio={compression_ratio:.1f}x | time={elapsed:.4f}s"
        )
        return result

# ==========================================
# MoE AGENT  —  Orchestrates all 7 pipeline stages
# ==========================================
class MoEAgent(BaseAIAgent):
    """
    Mixture of Experts Agent.

    Orchestrates the full MoE pipeline:
    Input → Router Mechanism → Parallel Experts → Top-k Selection → Advanced Patterning → Quantization → Output.

    Core principle: sparse activation (top-k of N). 
    The router dispatches to selected experts, they run in parallel, and a synthesis layer fuses the expert outputs into a final answer.
    """

    def __init__(self, client: Optional[OpenAI] = None) -> None:
        super().__init__(client)
        self._router = RouterMechanismStage()
        self._expert_inference = ExpertInferenceStage(self.client)
        self._top_k_selection = TopKSelectionStage()
        self._advanced_patterning = AdvancedPatterningStage()
        self._quantization = QuantizationStage()

    # public entry point
    def process(self, moe_input: MoEInput) -> MoEOutput:
        """
        Execute the full MoE pipeline.

        Args:
            moe_input: Validated MoEInput pydantic model.

        Returns:
            MoEOutput: Fully structured pipeline result with per-expert attribution.

        Raises:
            ValueError: On invalid input.
            RuntimeError: On unrecoverable API failure.
        """
        pipeline_start = time.perf_counter()
        self.logger.info(
            f"🚀 [MoE AGENT] Pipeline START | request_id={moe_input.request_id}"
        )
        self.logger.info(
            f"📥 [INPUT] query='{moe_input.query[:80]}...' | "
            f"top_k={moe_input.top_k} | "
            f"forced_experts={moe_input.require_experts or 'None'}"
        )
        try:
            # stage 2: router mechanism
            router = checkpointer.load("ROUTER", RouterResult)
            if not router:
                router = self._router.run(moe_input, self)
                checkpointer.save("ROUTER", router)
            else:
                logger.info(f"♻️ [CHECKPOINT] Loaded ROUTER from checkpoint: top_k={router.top_k_ids}")

            # stage 3: expert inference (parallel)
            expert_inference = checkpointer.load("EXPERT_INFERENCE", ExpertInferenceResult)
            if not expert_inference:
                expert_inference = self._expert_inference.run(router, moe_input.query, self._retry_api_call)
                checkpointer.save("EXPERT_INFERENCE", expert_inference)
            else:
                logger.info(f"♻️ [CHECKPOINT] Loaded EXPERT_INFERENCE from checkpoint: activated={expert_inference.activated_count}")

            # stage 4: top-k selection
            top_k_selection = checkpointer.load("TOP_K_SELECTION", TopKSelectionResult)
            if not top_k_selection:
                top_k_selection = self._top_k_selection.run(expert_inference, moe_input.top_k)
                checkpointer.save("TOP_K_SELECTION", top_k_selection)
            else:
                logger.info(f"♻️ [CHECKPOINT] Loaded TOP_K_SELECTION from checkpoint: k={top_k_selection.k}")

            # stage 5: advanced patterning
            advanced_patterning = checkpointer.load("ADVANCED_PATTERNING", AdvancedPatterningResult)
            if not advanced_patterning:
                advanced_patterning = self._advanced_patterning.run(top_k_selection, moe_input.query, self)
                checkpointer.save("ADVANCED_PATTERNING", advanced_patterning)
            else:
                logger.info(f"♻️ [CHECKPOINT] Loaded ADVANCED_PATTERNING from checkpoint: patterns={len(advanced_patterning.patterns)}")

            # stage 6: quantization
            quantization = checkpointer.load("QUANTIZATION", QuantizationResult)
            if not quantization:
                quantization = self._quantization.run(router)
                checkpointer.save("QUANTIZATION", quantization)
            else:
                logger.info(f"♻️ [CHECKPOINT] Loaded QUANTIZATION from checkpoint: bits={quantization.bits}")

            # stage 7: final output 
            expert_attribution = [
                f"{s.expert_id} ({s.expert_name}, weight={s.router_weight:.3f})"
                for s in top_k_selection.selected_experts
            ]
            total_time = time.perf_counter() - pipeline_start
            output = MoEOutput(
                request_id=moe_input.request_id,
                status=ProcessingStatus.SUCCESS,
                router=router,
                expert_inference=expert_inference,
                top_k_selection=top_k_selection,
                advanced_patterning=advanced_patterning,
                quantization=quantization,
                final_answer=advanced_patterning.synthesised_answer,
                expert_attribution=expert_attribution,
                total_pipeline_time=round(total_time, 4),
                metadata={
                    **moe_input.metadata,
                    'model': CHAT_MODEL,
                    'number_of_experts': NUM_EXPERTS,
                    'top_k': moe_input.top_k,
                }
            )
            logger.info(
                f"🎉 [MoE AGENT] Pipeline COMPLETE | total_time={total_time:.4f}s | "
                f"activated={expert_inference.activated_count}/{NUM_EXPERTS} | patterns={len(advanced_patterning.patterns)}"
            )
            logger.debug(
                f"🔍 [FINAL ANSWER] {output.final_answer[:300]} | "
                f"expert_attribution={output.expert_attribution}"
            )
            return output
        except Exception as e:
            elapsed = time.perf_counter() - pipeline_start
            logger.error(f"❌ [MoE AGENT] Pipeline FAILED after {elapsed:.4f}s | error={type(e).__name__}: {e}")
            raise RuntimeError("MoE pipeline execution failed.") from e

    def display(self, output: MoEOutput) -> None:
        div = "─" * 72
        print(f"\n{div}")
        print("  🟣 MoE AGENT — Mixture of Experts Pipeline Result")

        print(f"{div}")
        print(f"  Request ID       : {output.request_id}")
        print(f"  Status           : {output.status.value}")
        print(f"  Total Time       : {output.total_pipeline_time}s")
        
        print(f"{div}")
        print(f"  📤 FINAL ANSWER (fused from {len(output.expert_attribution)} experts)")
        print(f"\n  Attribution: {' + '.join(output.expert_attribution)}\n")
        print(f"  {output.final_answer}")
        print(f"\n{div}\n")
# ==========================================
# Instatiation
# ==========================================
def create_moe_agent(api_key: str = TOKEN, endpoint: str = ENDPOINT) -> MoEAgent:
    """Factory function to create an instance of MoEAgent with the provided API credentials."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    logger.info("🏭 [FACTORY] MoEAgent instantiated and ready.")
    return MoEAgent(client)

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    # create agent via factory function
    agent = create_moe_agent()

    # build validated input
    moe_input = MoEInput(
        query=(
            "What are the most significant trade-offs when scaling large language models beyond 100 billion parameters, and how should AI researchers prioritise compute, data quality, and architecture innovations?"
        ),
        top_k=2,  # sparse: only 2 of 4 experts activate
        require_experts=[],  # let the router decide freely
        metadata={"source": "moe_agent_demo", "version": "1.0"},
    )

    # execute the MoE pipeline
    result = agent.process(moe_input)

    # display the structured output
    agent.display(result)

    # export JSON (responses truncated for brevity) 
    # print("📦 Pydantic JSON snippet:")
    # print(result.model_dump_json(
    #     indent=2,
    #     exclude={
    #         "expert_inference": {
    #             "all_outputs": {"__all__": {"response"}}
    #         },
    #         "advanced_patterning": {"synthesised_answer": True},
    #         "top_k_selection": {
    #             "selected_experts": {"__all__": {"response"}}
    #         },
    #     }
    # ))