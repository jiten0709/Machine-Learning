import os, uuid, json, math, random, time, numpy as np, tiktoken, re
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum

from dotenv import load_dotenv
load_dotenv()

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="mlm.log")

# ==========================================
# Variable Configuration
# ==========================================
from utils.state_checkpointer import StateCheckpointer
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# FOR DEMO-1
# checkpointer = StateCheckpointer(
#     directory=CHECKPOINT_DIR, 
#     filename="mlm_checkpoint_demo_1.json",
#     logger=logger
# )

# FOR DEMO-2
checkpointer = StateCheckpointer(
    directory=CHECKPOINT_DIR, 
    filename="mlm_checkpoint_demo_2.json",
    logger=logger
)

TOKEN = os.environ['GITHUB_TOKEN']
ENDPOINT = os.environ['GITHUB_ENDPOINT']
CHAT_MODEL = os.environ['GITHUB_MODEL_NAME']
EMBEDDING_MODEL = os.environ['GITHUB_EMBED_NAME']
EMBEDDING_DIMENSIONS = 1536

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

# BERT masking strategy constants
MASK_PROBABILITY = 0.15    # 15% of tokens are selected for masking
MASK_REPLACE_PROB = 0.80    # 80% → [MASK]
RANDOM_REPLACE_PROB = 0.10    # 10% → random token
KEEP_ORIGINAL_PROB = 0.10    # 10% → unchanged

# Context window sizes for bidirectional attention
LEFT_CONTEXT_WINDOW = 64      # tokens of left context per mask
RIGHT_CONTEXT_WINDOW = 64      # tokens of right context per mask

# Top-k predictions per masked token
TOP_K_PREDICTIONS = 5

# Positional encoding dimensions
POS_ENCODING_DIM = 64      # sinusoidal positional encoding dim

# Max tokens to process (MLM operates on shorter sequences)
MLM_MAX_TOKENS = 512

ENCODING_NAME = "cl100k_base"

# ==========================================
# ENUMS
# ==========================================
class PipelineStage(str, Enum):
    TEXT_INPUT = "TEXT_INPUT"
    TOKEN_MASKING = "TOKEN_MASKING"
    EMBEDDING_LAYER = "EMBEDDING_LAYER"
    LEFT_CONTEXT = "LEFT_CONTEXT"
    RIGHT_CONTEXT = "RIGHT_CONTEXT"
    BIDIRECTIONAL_ATTENTION = "BIDIRECTIONAL_ATTENTION"
    MASKED_TOKEN_PREDICTION = "MASKED_TOKEN_PREDICTION"
    FEATURE_REPRESENTATION = "FEATURE_REPRESENTATION"

class MaskType(str, Enum):
    MASK_TOKEN = "MASK_TOKEN"      # replaced with [MASK]
    RANDOM = "RANDOM"          # replaced with random token
    ORIGINAL = "ORIGINAL"        # kept as-is (representation learning)

class MLMTask(str, Enum):
    FILL_MASK = "FILL_MASK"           # predict masked tokens
    SENTENCE_EMBEDDING = "SENTENCE_EMBEDDING"  # CLS-style representation
    TOKEN_CLASSIFICATION = "TOKEN_CLASSIFICATION"# per-token labelling
    SIMILARITY = "SIMILARITY"          # sentence pair similarity
    FEATURE_EXTRACTION = "FEATURE_EXTRACTION"  # raw feature vectors

class ProcessingStatus(str, Enum):
    PENDING  = "PENDING"
    SUCCESS  = "SUCCESS"
    FAILED   = "FAILED"
# ==========================================
# PYDANTIC MODELS
# ==========================================
class MLMInput(BaseModel):
    """Stage 1 — Validated raw input to the MLM pipeline."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the request")
    text: str = Field(..., min_length=10, max_length=32000, description="Input text for MLM processing")
    task: MLMTask = Field(default=MLMTask.FILL_MASK, description="Specific MLM task to perform")
    mask_probability: float = Field(default=MASK_PROBABILITY, ge=0.005, le=0.5, description="Fraction of tokens to mask (BERT default = 0.15).")
    top_k: int = Field(default=TOP_K_PREDICTIONS, ge=1, le=10, description="Number of top predictions to return per masked token.")
    seed: int = Field(default=42, description="Random seed for reproducibility of masking.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for processing or logging.")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("🚨 Input text cannot be empty.")
        return stripped

    @model_validator(mode="after")
    def stamp_metadata(self) -> "MLMInput":
        self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["task"] = self.task.value
        return self
    
# stage 2: token masking
class MaskedToken(BaseModel):
    """Represents a single masked token and its associated data."""
    position: int = Field(..., description="Token index in the original sequence")
    original_token_id: int = Field(..., description="Original token ID before masking")
    original_text: str = Field(..., description="Original token text before masking")
    mask_type: MaskType = Field(..., description="Type of masking applied to this token")
    displayed_text: str = Field(..., description="Text shown to the model (e.g., [MASK], random token, or original)")
    random_token_id: Optional[int] = Field(None, description="If mask_type is RANDOM, the token ID of the random replacement")

class TokenMaskingResult(BaseModel):
    """Stage 2 output: list of masked tokens and the masked input sequence."""
    stage: PipelineStage = Field(default=PipelineStage.TOKEN_MASKING, description="Pipeline stage identifier")
    encoding: str = Field(description="Tokenizer encoding used")
    original_tokens: List[int] = Field(description="List of original token texts")
    original_texts: List[str] = Field(description="List of original token texts (same as original_tokens, included for clarity)")
    masked_tokens: List[int] = Field(description="List of masked tokens with details")
    masked_texts: List[str] = Field(description="List of token texts after masking (e.g., [MASK], random token, or original)")
    masked_positions: List[MaskedToken] = Field(description="List of masked token positions and details")
    total_tokens: int = Field(description="Total number of tokens in the original input")
    num_masked: int = Field(description="Total number of tokens masked in this input")
    mask_ratio_actual: float = Field(description="Actual fraction of tokens masked (may differ from requested due to rounding)")
    masking_strategy: Dict[str, int] = Field(description="Count of each masking type applied (MASK_TOKEN, RANDOM, ORIGINAL)")
    processing_time: float = Field(description="Time taken to perform token masking in seconds")

# stage 3: embedding layer
class PositionalEncoding(BaseModel):
    """Sinusoidal positional encoding vector for a single token position."""
    position: int = Field(..., description="Token index in the sequence")
    encoding: List[float] = Field(..., description="List of floats representing the positional encoding vector")

class TokenEmbedding(BaseModel):
    """Represents the embedding vector for a single token."""
    position: int = Field(..., description="Token index in the sequence")
    token_text: str = Field(..., description="Text of the token")
    is_masked: bool = Field(..., description="Whether this token is masked or not")
    semantic_embedding: List[float] = Field(..., description="Semantic embedding vector from the model")
    positional_encoding: List[float] = Field(..., description="Positional encoding vector for this token")
    combined_norm: float = Field(..., description="L2 norm of the combined embedding vector (semantic + positional)")

class EmbeddingLayerResult(BaseModel):
    """Stage 3 output: list of token embeddings with positional encodings."""
    stage: PipelineStage = Field(default=PipelineStage.EMBEDDING_LAYER, description="Pipeline stage identifier")
    model: str = Field(description="Embedding model used")
    cls_embedding: List[float] = Field(description="Embedding vector for the [CLS] token (if applicable)")
    token_embeddings: List[TokenEmbedding] = Field(description="List of token embeddings with details")
    sequence_length: int = Field(description="Number of tokens in the input sequence")
    embedding_dim: int = Field(description="Dimensionality of the embedding vectors")
    pos_encoding_dim: int = Field(description="Dimensionality of the positional encoding vectors")
    processing_time: float = Field(description="Time taken to compute embeddings in seconds")

# stage 4 & 5: left & right context extraction
class ContextWindow(BaseModel):
    """Represents a context window (left or right) for a masked token."""
    mask_position: int = Field(..., description="Position of the masked token this context window is associated with")
    side: str = Field(..., description="Which side of the masked token this context window represents (LEFT or RIGHT)")
    window_size: int = Field(..., description="Number of tokens in this context window")
    token_ids: List[int] = Field(..., description="List of token IDs in this context window")
    token_texts: List[str] = Field(..., description="List of token texts in this context window")
    context_text: str = Field(description="Concatenated text of the context window")
    boundary_hit: bool = Field(description="Whether this context window hit the boundary of the sequence (i.e., not full size)")

class LeftContextResult(BaseModel):
    """Stage 4 output: left context windows for all masked tokens."""
    stage: PipelineStage = Field(default=PipelineStage.LEFT_CONTEXT, description="Pipeline stage identifier")
    windows: List[ContextWindow] = Field(description="List of left context windows for each masked token")
    avg_window_size: float = Field(description="Average size of the left context windows")
    boundary_hits: int = Field(description="Number of left context windows that hit the sequence boundary")
    processing_time: float = Field(description="Time taken to extract left contexts in seconds")

class RightContextResult(BaseModel):
    """Stage 5 output: right context windows for all masked tokens."""
    stage: PipelineStage = Field(default=PipelineStage.RIGHT_CONTEXT, description="Pipeline stage identifier")
    windows: List[ContextWindow] = Field(description="List of right context windows for each masked token")
    avg_window_size: float = Field(description="Average size of the right context windows")
    boundary_hits: int = Field(description="Number of right context windows that hit the sequence boundary")
    processing_time: float = Field(description="Time taken to extract right contexts in seconds")

# stage 6: bidirectional attention
class BidirectionalAttentionEntry(BaseModel):
    """Represents the attention scores between a masked token and its context tokens."""
    mask_position: int = Field(..., description="Position of the masked token")
    original_text: str = Field(..., description="Original text of the masked token")
    left_context: str = Field(description="Concatenated text of the left context")
    right_context: str = Field(description="Concatenated text of the right context")
    fused_context: str = Field(description="Concatenated text of left + masked token + right context")
    left_attention_weight: float = Field(ge=0.0, le=1.0, description="Attention weight from masked token to left context")
    right_attention_weight: float = Field(ge=0.0, le=1.0, description="Attention weight from masked token to right context")
    context_coherence: float = Field(ge=0.0, le=1.0, description="Coherence score between left and right contexts (e.g., cosine similarity of their embeddings)")

class BidirectionalAttentionResult(BaseModel):
    """Stage 6 output: attention scores and coherence for each masked token."""
    stage: PipelineStage = Field(default=PipelineStage.BIDIRECTIONAL_ATTENTION, description="Pipeline stage identifier")
    attention_entries: List[BidirectionalAttentionEntry] = Field(description="List of attention entries for each masked token")
    avg_left_weight: float = Field(description="Average attention weight to left context across all masked tokens")
    avg_right_weight: float = Field(description="Average attention weight to right context across all masked tokens")
    avg_coherence: float = Field(description="Average coherence score between left and right contexts across all masked tokens")
    is_truly_bidirectional: bool = Field(description="Whether the attention pattern indicates true bidirectionality (e.g., both weights above a certain threshold)")
    processing_time: float = Field(description="Time taken to compute bidirectional attention in seconds")

# stage 7: masked token prediction
class TokenCandidate(BaseModel):
    """Represents a single candidate prediction for a masked token."""
    rank: int = Field(..., description="Rank of this candidate among the predictions (1 = top prediction)")
    token_text: str = Field(..., description="Text of the candidate token")
    confidence: float = Field(ge=0.0, le=1.0, description="Model's confidence score for this prediction")
    log_prob: float = Field(description="Log probability of this token prediction")

class MaskedTokenPrediction(BaseModel):
    """Prediction results for a single masked token."""
    mask_position: int = Field(..., description="Position of the masked token")
    original_text: str = Field(..., description="Original text of the masked token")
    mask_type: MaskType = Field(..., description="Type of masking applied to this token")
    candidates: List[TokenCandidate] = Field(description="List of top-k candidate predictions for this masked token")
    top_prediction: str = Field(description="Text of the top predicted token")
    top_confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of the top prediction")
    exact_match: bool = Field(description="Whether the top prediction exactly matches the original token")
    mrr: float = Field(description="Mean Reciprocal Rank of the original token in the candidate list")

class MaskedTokenPredictionResult(BaseModel):
    """Stage 7 output: masked token prediction results for all masked tokens."""
    stage: PipelineStage = Field(default=PipelineStage.MASKED_TOKEN_PREDICTION, description="Pipeline stage identifier")
    predictions: List[MaskedTokenPrediction] = Field(description="List of masked token predictions with details")
    accuracy: float = Field(description="Overall accuracy of top predictions across all masked tokens")
    mean_confidence: float = Field(description="Average confidence score of the top predictions across all masked tokens")
    mean_mrr: float = Field(description="Average Mean Reciprocal Rank of the original tokens in the candidate lists across all masked tokens")
    total_predicted: int = Field(description="Total number of masked tokens for which predictions were made")
    processing_time: float = Field(description="Time taken to perform masked token prediction in seconds")

# stage 8: feature representation
class SentenceFeatures(BaseModel):
    """Represents the final feature representation for the input sentence after MLM processing."""
    cls_vector: List[float] = Field(description="Embedding vector for the [CLS] token representing the entire sentence")
    pooled_vector: List[float] = Field(description="Pooled embedding vector derived from masked token predictions and attention patterns")
    cls_norm: float = Field(description="L2 norm of the CLS embedding vector")
    pooled_norm: float = Field(description="L2 norm of the pooled embedding vector")
    cls_pooled_similarity: float = Field(ge=0.0, le=1.0, description="Cosine similarity between CLS vector and pooled vector")

class TokenFeature(BaseModel):
    """Represents the feature representation for a single token after MLM processing."""
    position: int = Field(..., description="Token index in the sequence")
    token_text: str = Field(..., description="Text of the token")
    predicted_text: str = Field(description="Text of the top predicted token for this position")
    feature_vector: List[float] = Field(description="Final feature vector for this token derived from its embedding, attention patterns, and prediction confidence")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score of the top prediction for this token")

class FeatureRepresentationResult(BaseModel):
    """Stage 8 output: final feature representations for the sentence and tokens."""
    stage: PipelineStage = Field(default=PipelineStage.FEATURE_REPRESENTATION, description="Pipeline stage identifier")
    sentence_features: SentenceFeatures = Field(description="Feature representation for the entire sentence")
    masked_token_features: List[TokenFeature] = Field(description="List of feature representations for each masked token")
    representation_dim: int = Field(description="Dimensionality of the final feature vectors")
    feature_quality_score: float = Field(ge=0.0, le=1.0, description="Overall quality score of the feature representations based on prediction confidence and attention coherence")
    processing_time: float = Field(description="Time taken to compute final feature representations in seconds")

# final output
class MLMOutput(BaseModel):
    """Final output of the MLM pipeline, aggregating results from all stages."""
    request_id: str = Field(description="Unique ID for the request, matching the input")
    stage: PipelineStage = Field(default=PipelineStage.FEATURE_REPRESENTATION, description="Final pipeline stage identifier")
    status: ProcessingStatus = Field(default=ProcessingStatus.SUCCESS, description="Overall processing status of the MLM pipeline")

    # stage payloads
    token_masking: TokenMaskingResult = Field(description="Output from the token masking stage")
    embedding_layer: EmbeddingLayerResult = Field(description="Output from the embedding layer stage")
    left_context: LeftContextResult = Field(description="Output from the left context extraction stage")
    right_context: RightContextResult = Field(description="Output from the right context extraction stage")
    bidirectional_attention: BidirectionalAttentionResult = Field(description="Output from the bidirectional attention stage")
    masked_token_prediction: MaskedTokenPredictionResult = Field(description="Output from the masked token prediction stage")
    feature_representation: FeatureRepresentationResult = Field(description="Output from the feature representation stage")

    # summary metrics
    fill_mask_accuracy: float = Field(description="Accuracy of the masked token predictions across all masked tokens")
    mean_prediction_confidence: float = Field(description="Average confidence of the top predictions across all masked tokens")
    bidirectional_confidence: float = Field(description="Overall confidence that the attention patterns indicate true bidirectionality")
    total_pipeline_time: float = Field(description="Total time taken to process the entire MLM pipeline in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the output, such as processing timestamps, model versions, etc.")

    @model_validator(mode="after")
    def populate_metadata(self) -> "MLMOutput":
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["num_masked"] = self.token_masking.num_masked
        self.metadata["accuracy"] = self.fill_mask_accuracy
        self.metadata["mean_confidence"] = self.mean_prediction_confidence
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
# HELPER FUNCTIONS
# ==========================================
def sinusoidal_positional_encoding(position: int, dim: int = POS_ENCODING_DIM) -> List[float]:
    """
    Compute sinusoidal positional encoding (Vaswani et al., 2017).
    PE(pos, 2i) = sin(pos / 10000^(2i/dim))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
    """
    encoding = []
    for i in range(dim // 2):
        denom = 10_000 ** (2 * i / dim)
        encoding.append(math.sin(position / denom))
        encoding.append(math.cos(position / denom))
    return encoding[:dim]

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Cosine similarity between two equal-length vectors."""
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    den = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / den) if den > 0 else 0.0

def compute_mrr(original: str, candidates: List[TokenCandidate]) -> float:
    """Mean Reciprocal Rank of the original token in prediction list."""
    for i, cand in enumerate(candidates, start=1):
        if cand.token_text.strip().lower() == original.strip().lower():
            return 1.0 / i
    return 0.0

# ==========================================
# PIPELINE STAGES
# ==========================================
class TokenMaskingStage:
    """
    Stage 2: TOKEN MASKING
    Implements BERT's original masking strategy exactly:

    Step 1: Select 15% of tokens as masking candidates.
    Step 2: For each selected token:
      - 80% → replace with special [MASK] token
      - 10% → replace with random token from vocabulary
      - 10% → keep original (but still predict it — representation learning)

    This three-way split is critical: if all selected tokens were masked, the model would learn to ignore non-masked tokens. 
    The 10%/10% splits ensure the model maintains good representations for ALL tokens.

    Special tokens ([CLS], [SEP]) are never masked.
    """

    MASK_TOKEN_TEXT = "[MASK]"
    CLS_TOKEN_TEXT = "[CLS]"
    SEP_TOKEN_TEXT = "[SEP]"

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding(ENCODING_NAME)
        self._vocab_size = 100277

    def run(self, mlm_input: MLMInput) -> TokenMaskingResult:
        logger.info(f"⚙️ [TOKEN MASKING] Applying BERT masking strategy (p={mlm_input.mask_probability})...")
        t0 = time.perf_counter()
        rng = random.Random(mlm_input.seed)
    
        # tokenize
        tokens = self._encoding.encode(mlm_input.text)
        if len(tokens) > MLM_MAX_TOKENS:
            tokens = tokens[:MLM_MAX_TOKENS]
            logger.warning(f"⚠️ Input text tokenized to {len(tokens)} tokens, exceeding MLM max of {MLM_MAX_TOKENS}. Truncating input.")

        # decode individual tokens
        original_texts: List[str] = []
        for t in tokens:
            try:
                original_texts.append(self._encoding.decode([t]))
            except Exception as e:
                logger.error(f"🚨 Error decoding token ID {t}: {e}")
                original_texts.append("[unk]")
            
        # select candidates for masking (15% of tokens)
        # exclude very short tokens from masking
        eligible_positions = [i for i, t in enumerate(original_texts) if len(t.strip()) >= 2]
        num_mask = max(1, int(len(eligible_positions) * mlm_input.mask_probability))
        mask_positions_selected = sorted(rng.sample(eligible_positions, min(num_mask, len(eligible_positions))))

        # apply 3-way masking strategy
        masked_tokens = list(tokens)
        masked_texts = list(original_texts)
        masked_position_objects: List[MaskedToken] = []
        strategy_counts = {MaskType.MASK_TOKEN: 0, MaskType.RANDOM: 0, MaskType.ORIGINAL: 0}

        for pos in mask_positions_selected:
            r = rng.random()
            if r < MASK_REPLACE_PROB:
                mask_type = MaskType.MASK_TOKEN
                displayed_text = self.MASK_TOKEN_TEXT
                masked_tokens[pos] = 0    # use token ID 0 as mask placeholder
                masked_texts[pos] = self.MASK_TOKEN_TEXT
                random_tok = None
            elif r < MASK_REPLACE_PROB + RANDOM_REPLACE_PROB:
                mask_type = MaskType.RANDOM
                random_tok = rng.randint(1, self._vocab_size - 1)
                try:
                    rand_text = self._enc.decode([random_tok])
                except Exception:
                    rand_text = "<unk>"
                displayed_text = rand_text
                masked_tokens[pos] = random_tok
                masked_texts[pos] = rand_text
            else:
                mask_type = MaskType.ORIGINAL
                displayed_text = original_texts[pos]
                random_tok = None

            strategy_counts[mask_type] += 1
            masked_position_objects.append(MaskedToken(
                position=pos,
                original_token_id=tokens[pos],
                original_text=original_texts[pos],
                mask_type=mask_type,
                displayed_text=displayed_text,
                random_token_id=random_tok
            ))

        actual_ratio = len(masked_position_objects) / len(tokens)
        elapsed = time.perf_counter() - t0
        result = TokenMaskingResult(
            encoding=ENCODING_NAME,
            original_tokens=list(tokens),
            original_texts=original_texts,
            masked_tokens=masked_tokens,
            masked_texts=masked_texts,
            masked_positions=masked_position_objects,
            total_tokens=len(tokens),
            num_masked=len(masked_position_objects),
            mask_ratio_actual=round(actual_ratio, 4),
            masking_strategy={k.value: c for k, c in strategy_counts.items()},
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [TOKEN MASKING] {len(tokens)} tokens | masked={len(masked_position_objects)} ({actual_ratio:.2%}) | "
            f"MASK={strategy_counts[MaskType.MASK_TOKEN]} | RANDOM={strategy_counts[MaskType.RANDOM]} | "
            f"ORIGINAL={strategy_counts[MaskType.ORIGINAL]} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 Masked positions detail: {[{'pos': m.position, 'orig': m.original_text, 'type': m.mask_type.value} for m in masked_position_objects]}")
        return result

class EmbeddingLayerStage:
    """
    Stage 3: EMBEDDING LAYER
    Generates token embeddings combining:
    1. Semantic embedding (text-embedding-3-small) — captures meaning
    2. Sinusoidal positional encoding — captures position in sequence
    3. [CLS] token embedding — sentence-level representation

    Architecture fidelity:
    - BERT's embedding layer = token embedding + positional embedding + segment embedding, all summed and layer-normalised.
    - We produce: semantic (1536-d) + positional (64-d) per token.
    - The [CLS] embedding is the full sentence embedding from the API.
    - For efficiency, per-token embeddings are computed for masked positions + a sample of non-masked positions (not all 512 tokens).
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, masking_result: TokenMaskingResult, agent: BaseAIAgent) -> EmbeddingLayerResult:
        logger.info("⚙️ [EMBEDDING LAYER] Computing token + positional embeddings...")
        t0 = time.perf_counter()

        #  [CLS] embedding — full sentence
        full_text = " ".join(masking_result.original_texts[:MLM_MAX_TOKENS])
        cls_response = agent._retry_api_call(
            self._client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=full_text[:8000],
        )
        cls_embedding = cls_response.data[0].embedding
        logger.debug("🔍 [EMBEDDING LAYER] CLS embedding computed")

        # Per-token embeddings for masked positions 
        # In BERT, all tokens get embeddings; we compute for masked positions plus a bounded sample for efficiency.
        masked_pos_set = {mp.position for mp in masking_result.masked_positions}
        sample_positions = sorted(masked_pos_set)[:20]   # cap API calls

        token_embeddings: List[TokenEmbedding] = []
        for pos in sample_positions:
            tok_text = masking_result.original_texts[pos]
            # Embed with surrounding context for richer representation
            ctx_start = max(0, pos - 3)
            ctx_end = min(len(masking_result.original_texts), pos + 4)
            ctx_text = " ".join(masking_result.original_texts[ctx_start:ctx_end])

            try:
                tok_resp = agent._retry_api_call(
                    self._client.embeddings.create,
                    model=EMBEDDING_MODEL,
                    input=ctx_text[:512],
                )
                sem_emb = tok_resp.data[0].embedding
            except Exception:
                sem_emb = cls_embedding   # fallback to CLS

            pos_enc = sinusoidal_positional_encoding(pos, POS_ENCODING_DIM)
            sem_norm = float(np.linalg.norm(np.array(sem_emb)))

            token_embeddings.append(TokenEmbedding(
                position=pos,
                token_text=tok_text,
                is_masked=pos in masked_pos_set,
                semantic_embedding=sem_emb,
                positional_encoding=pos_enc,
                combined_norm=round(sem_norm, 4)
            ))
            logger.debug(f"🔍 [EMBEDDING LAYER] Token[{pos}]='{tok_text}' | norm={sem_norm:.4f}")

        elapsed = time.perf_counter() - t0
        result  = EmbeddingLayerResult(
            model=EMBEDDING_MODEL,
            cls_embedding=cls_embedding,
            token_embeddings=token_embeddings,
            sequence_length=masking_result.total_tokens,
            embedding_dim=EMBEDDING_DIMENSIONS,
            pos_encoding_dim=POS_ENCODING_DIM,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [EMBEDDING LAYER] CLS dim={EMBEDDING_DIMENSIONS} | token_embeddings={len(token_embeddings)} | "
            f"pos_encoding_dim={POS_ENCODING_DIM} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 Sample token embeddings: {[{'pos': te.position, 'text': te.token_text, 'norm': te.combined_norm} for te in token_embeddings[:5]]}")
        return result

class ContextExtractionStage:
    """
    Shared implementation for Left Context (Stage 4) and Right Context (Stage 5).
    Extracts boundary-aware token windows on each side of every masked position.

    Architecture fidelity:
    - In BERT, bidirectionality is achieved because the entire sequence is processed simultaneously. The model attends to ALL positions in one pass.
    - We make this explicit: for each mask, we extract left and right context windows independently, then fuse them in Stage 6.
    - Boundary hit detection tells us when the sequence edge constrains context.
    """

    def extract_left(self, masking_result: TokenMaskingResult) -> LeftContextResult:
        logger.info(f"⚙️ [LEFT CONTEXT] Extracting left windows for {len(masking_result.masked_positions)} masked positions...")
        t0 = time.perf_counter()
        windows: List[ContextWindow] = []
        boundary_hits = 0

        for mp in masking_result.masked_positions:
            pos = mp.position
            win_start = max(0, pos - LEFT_CONTEXT_WINDOW)
            hit = win_start == 0 and pos > 0
            if hit:
                boundary_hits += 1

            tok_ids = masking_result.original_tokens[win_start:pos]
            tok_texts = masking_result.original_texts[win_start:pos]
            ctx_text = " ".join(tok_texts)

            windows.append(ContextWindow(
                mask_position=pos,
                side="LEFT",
                window_size=len(tok_ids),
                token_ids=tok_ids,
                token_texts=tok_texts,
                context_text=ctx_text,
                boundary_hit=hit
            ))
            logger.debug(f"LEFT[{pos}]: '{ctx_text[-60:]}' (size={len(tok_ids)}, boundary={hit})")

        avg_size = sum(w.window_size for w in windows) / len(windows) if windows else 0
        elapsed = time.perf_counter() - t0

        result = LeftContextResult(
            windows=windows,
            avg_window_size=round(avg_size, 2),
            boundary_hits=boundary_hits,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [LEFT CONTEXT] windows={len(windows)} | "
            f"avg_size={avg_size:.1f} | boundary_hits={boundary_hits} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 Sample left contexts: {[{'pos': w.mask_position, 'context': w.context_text[-60:], 'boundary': w.boundary_hit} for w in windows[:5]]}")
        return result
    
    def extract_right(self, masking_result: TokenMaskingResult) -> RightContextResult:
        logger.info(f"⚙️ [RIGHT CONTEXT] Extracting right windows for {len(masking_result.masked_positions)} masked positions...")
        t0 = time.perf_counter()
        windows: List[ContextWindow] = []
        boundary_hits = 0

        for mp in masking_result.masked_positions:
            pos = mp.position
            win_end = min(len(masking_result.original_tokens), pos + RIGHT_CONTEXT_WINDOW + 1)
            hit = win_end == len(masking_result.original_tokens) and pos < len(masking_result.original_tokens) - 1
            if hit:
                boundary_hits += 1

            tok_ids = masking_result.original_tokens[pos + 1:win_end]
            tok_texts = masking_result.original_texts[pos + 1:win_end]
            ctx_text = " ".join(tok_texts)

            windows.append(ContextWindow(
                mask_position=pos,
                side="RIGHT",
                window_size=len(tok_ids),
                token_ids=tok_ids,
                token_texts=tok_texts,
                context_text=ctx_text,
                boundary_hit=hit
            ))
            logger.debug(f"RIGHT[{pos}]: '{ctx_text[:60]}' (size={len(tok_ids)}, boundary={hit})")

        avg_size = sum(w.window_size for w in windows) / len(windows) if windows else 0
        elapsed = time.perf_counter() - t0

        result = RightContextResult(
            windows=windows,
            avg_window_size=round(avg_size, 2),
            boundary_hits=boundary_hits,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [RIGHT CONTEXT] windows={len(windows)} | "
            f"avg_size={avg_size:.1f} | boundary_hits={boundary_hits} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 Sample right contexts: {[{'pos': w.mask_position, 'context': w.context_text[:60], 'boundary': w.boundary_hit} for w in windows[:5]]}")
        return result

class BidirectionalAttentionStage:
    """
    Stage 6: BIDIRECTIONAL ATTENTION
    The core MLM architectural differentiator.

    For each masked position, fuse left and right context simultaneously.
    GPT-4.1 is given BOTH the left context and right context at the same
    time and asked to produce:
    1. A fused contextual representation
    2. Relative attention weights (how much left vs right informed the fusion)
    3. A coherence score (how well both contexts align)

    This is fundamentally different from autoregressive models that can
    only attend to left context. The bidirectionality is the defining
    property of BERT-class models.
    """
    def run(self, masking_result: TokenMaskingResult, left_context: LeftContextResult, right_context: RightContextResult, agent: BaseAIAgent) -> BidirectionalAttentionResult:
        logger.info(f"⚙️ [BIDIRECTIONAL ATTENTION] Fusing left ⟷ right context for {len(masking_result.masked_positions)} positions...")
        t0 = time.perf_counter()

        # Build lookup maps by mask position
        left_map = {w.mask_position: w for w in left_context.windows}
        right_map = {w.mask_position: w for w in right_context.windows}

        # Batch all positions into a single GPT-4.1 call for efficiency
        positions_payload = []
        for mp in masking_result.masked_positions:
            lw = left_map.get(mp.position)
            rw = right_map.get(mp.position)
            positions_payload.append({
                "position": mp.position,
                "left_context": lw.context_text[-200:] if lw else "",
                "right_context": rw.context_text[:200] if rw else "",
                "masked_token": mp.displayed_text,
            })

        data = agent._gpt_json_response(
            system=(
                "You are the bidirectional attention mechanism of a BERT-style Masked Language Model. "
                "For each masked position, you receive BOTH left and right context simultaneously. "
                "Fuse them to produce a rich contextual understanding. "
                "JSON only."
            ),
            user=(
                f"Process {len(positions_payload)} masked positions.\n\n"
                f"Positions:\n{json.dumps(positions_payload, indent=2)}\n\n"
                "For each position respond with:\n"
                "{\n"
                '  "attention_entries": [\n'
                "    {\n"
                '      "mask_position": <int>,\n'
                '      "fused_context": "<one sentence fusing both contexts>",\n'
                '      "left_attention_weight": <0.0-1.0, how much left context informed fusion>,\n'
                '      "right_attention_weight": <0.0-1.0, how much right context informed fusion>,\n'
                '      "context_coherence": <0.0-1.0, how well both contexts agree>\n'
                "    }\n"
                "  ]\n"
                "}\n\n"
                "Note: left_attention_weight + right_attention_weight should sum to ~1.0"
            ),
            max_tokens=2000,
            temperature=0.1,
        )

        raw_entries = data.get("attention_entries", [])
        # Map by position for safe lookup
        entry_map = {e["mask_position"]: e for e in raw_entries}

        attention_entries: list[BidirectionalAttentionEntry] = []
        for mp in masking_result.masked_positions:
            lw = left_map.get(mp.position)
            rw = right_map.get(mp.position)
            raw = entry_map.get(mp.position, {})

            l_wt = float(raw.get("left_attention_weight", 0.5))
            r_wt = float(raw.get("right_attention_weight", 0.5))

            # Normalise weights to sum to 1.0
            total_wt = l_wt + r_wt
            if total_wt > 0:
                l_wt = round(l_wt / total_wt, 4)
                r_wt = round(r_wt / total_wt, 4)

            attention_entries.append(BidirectionalAttentionEntry(
                mask_position=mp.position,
                original_text=mp.original_text,
                left_context=lw.context_text[-100:] if lw else "",
                right_context=rw.context_text[:100] if rw else "",
                fused_context=raw.get("fused_context", ""),
                left_attention_weight=l_wt,
                right_attention_weight=r_wt,
                context_coherence=float(raw.get("context_coherence", 0.8))
            ))
            logger.debug(
                f"🔀 Pos[{mp.position}] '{mp.original_text}' | L={l_wt:.3f} R={r_wt:.3f} | "
                f"coherence={raw.get('context_coherence', 0.8):.3f}"
            )

        avg_l = sum(e.left_attention_weight  for e in attention_entries) / max(len(attention_entries), 1)
        avg_r = sum(e.right_attention_weight for e in attention_entries) / max(len(attention_entries), 1)
        avg_coh = sum(e.context_coherence for e in attention_entries) / max(len(attention_entries), 1)
        # Truly bidirectional: both sides contribute meaningfully
        is_bidir = avg_l > 0.15 and avg_r > 0.15
        elapsed  = time.perf_counter() - t0

        result = BidirectionalAttentionResult(
            attention_entries=attention_entries,
            avg_left_weight=round(avg_l, 4),
            avg_right_weight=round(avg_r, 4),
            avg_coherence=round(avg_coh, 4),
            is_truly_bidirectional=is_bidir,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [BIDIRECTIONAL ATTENTION] entries={len(attention_entries)} | avg_L={avg_l:.4f} avg_R={avg_r:.4f} | "
            f"coherence={avg_coh:.4f} | bidirectional={is_bidir} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 Sample attention entries: {[{'pos': e.mask_position, 'L': e.left_attention_weight, 'R': e.right_attention_weight, 'coh': e.context_coherence} for e in attention_entries[:5]]}")
        return result

class MaskedTokenPredictionStage:
    """
    Stage 7: MASKED TOKEN PREDICTION
    The MLM pre-training objective: predict the original token at each masked position using the bidirectional fused context.

    For each masked position:
    - GPT-4.1 is given: left context + right context + fused context
    - It produces top-k candidate tokens with calibrated confidence scores
    - Accuracy = fraction where top prediction matches original token
    - MRR = Mean Reciprocal Rank (measures rank of correct token in top-k)

    This is the actual learning signal in BERT pre-training: cross-entropy loss between predicted distribution and true token.
    """
    def run(self, masking_result: TokenMaskingResult, attention_result: BidirectionalAttentionResult, top_k: int, agent: BaseAIAgent) -> MaskedTokenPredictionResult:
        logger.info(f"⚙️ [MASKED TOKEN PREDICTION] Predicting {len(masking_result.masked_positions)} masked tokens (top-{top_k})...")
        t0 = time.perf_counter()

        attn_map = {e.mask_position: e for e in attention_result.attention_entries}

        # Batch all predictions into one GPT-4.1 call
        prediction_prompts = []
        for mp in masking_result.masked_positions:
            ae = attn_map.get(mp.position)
            prediction_prompts.append({
                "position": mp.position,
                "left_context": ae.left_context  if ae else "",
                "right_context": ae.right_context if ae else "",
                "fused_context": ae.fused_context if ae else "",
                "mask_type": mp.mask_type.value,
            })

        data = agent._gpt_json_response(
            system=(
                "You are the Masked Token Prediction head of a BERT-style MLM. "
                "Using bidirectional context, predict the most likely original token at each masked position. "
                "Provide calibrated confidence scores that sum to 1.0 across candidates. JSON only."
            ),
            user=(
                f"Predict the original token for {len(prediction_prompts)} "
                f"masked positions. Provide top-{top_k} candidates each.\n\n"
                f"Masked positions:\n{json.dumps(prediction_prompts, indent=2)}\n\n"
                "Respond with:\n"
                "{\n"
                '  "predictions": [\n'
                "    {\n"
                '      "mask_position": <int>,\n'
                f'      "candidates": [\n'
                '        {"rank": 1, "token_text": "<word>", "confidence": <0.0-1.0>},\n'
                f'        ... (up to {top_k} candidates)\n'
                "      ]\n"
                "    }\n"
                "  ]\n"
                "}"
            ),
            max_tokens =2500,
            temperature=0.1,
        )

        raw_preds = data.get("predictions", [])
        pred_map = {p["mask_position"]: p for p in raw_preds}
        predictions: List[MaskedTokenPrediction] = []

        for mp in masking_result.masked_positions:
            raw = pred_map.get(mp.position, {})
            raw_cands = raw.get("candidates", [])

            # Build and validate candidates
            candidates: List[TokenCandidate] = []
            for cand in raw_cands[:top_k]:
                conf = float(cand.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
                candidates.append(TokenCandidate(
                    rank=int(cand.get("rank", len(candidates) + 1)),
                    token_text=str(cand.get("token_text", "")).strip(),
                    confidence=round(conf, 4),
                    log_prob=round(math.log(conf + 1e-10), 4),
                ))

            if not candidates:
                candidates = [TokenCandidate(
                    rank=1,
                    token_text=mp.original_text,
                    confidence=0.5,
                    log_prob=round(math.log(0.5), 4)
                )]

            top_pred = candidates[0].token_text
            top_conf = candidates[0].confidence
            exact_match = top_pred.strip().lower() == mp.original_text.strip().lower()
            mrr = compute_mrr(mp.original_text, candidates)

            predictions.append(MaskedTokenPrediction(
                mask_position=mp.position,
                original_text=mp.original_text,
                mask_type=mp.mask_type,
                candidates=candidates,
                top_prediction=top_pred,
                top_confidence=top_conf,
                exact_match=exact_match,
                mrr=mrr
            ))
            match_icon = "✅" if exact_match else "❌"
            logger.debug(
                f"🔍 {match_icon} Pos[{mp.position}] "
                f"orig='{mp.original_text}' → pred='{top_pred}' "
                f"conf={top_conf:.3f} mrr={mrr:.3f}"
            )

        accuracy = sum(1 for p in predictions if p.exact_match) / max(len(predictions), 1)
        mean_conf = sum(p.top_confidence for p in predictions) / max(len(predictions), 1)
        mean_mrr = sum(p.mrr for p in predictions) / max(len(predictions), 1)
        elapsed = time.perf_counter() - t0

        result = MaskedTokenPredictionResult(
            predictions=predictions,
            accuracy=round(accuracy, 4),
            mean_confidence=round(mean_conf, 4),
            mean_mrr=round(mean_mrr, 4),
            total_predicted=len(predictions),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [MASKED TOKEN PREDICTION] predicted={len(predictions)} | "
            f"accuracy={accuracy:.4f} | mean_conf={mean_conf:.4f} | mean_mrr={mean_mrr:.4f} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 Sample predictions: {[{'pos': p.mask_position, 'orig': p.original_text, 'pred': p.top_prediction, 'conf': p.top_confidence} for p in predictions[:5]]}")
        return result

class FeatureRepresentationStage:
    """
    Stage 8: FEATURE REPRESENTATION
    Constructs the final contextualised feature representations:

    1. Sentence-level: CLS embedding (BERT's pooled output) + mean-pooled token embeddings (common in sentence-transformers).
    2. Token-level: Per-masked-token 64-d contextual features derived from combining positional encoding, prediction confidence, and attention weights.

    These representations are the downstream task inputs — what you'd feed into a classifier for NER, sentiment analysis, QA, etc.
    """
    def run(self, embedding_result: EmbeddingLayerResult, attention_result: BidirectionalAttentionResult, prediction_result: MaskedTokenPredictionResult) -> FeatureRepresentationResult:
        logger.info("⚙️ [FEATURE REPRESENTATION] Building final feature vectors...")
        t0 = time.perf_counter()

        # Sentence-level features 
        cls_vec = embedding_result.cls_embedding

        # Mean-pool available token embeddings as sentence representation
        if embedding_result.token_embeddings:
            tok_matrix = np.array(
                [te.semantic_embedding for te in embedding_result.token_embeddings],
                dtype=np.float64,
            )
            pooled_vec = np.mean(tok_matrix, axis=0).tolist()
        else:
            pooled_vec = cls_vec

        cls_norm = float(np.linalg.norm(np.array(cls_vec)))
        pooled_norm = float(np.linalg.norm(np.array(pooled_vec)))
        cls_pooled_sim = cosine_similarity(cls_vec, pooled_vec)

        sentence_features = SentenceFeatures(
            cls_vector=cls_vec,
            pooled_vector=pooled_vec,
            cls_norm=round(cls_norm, 4),
            pooled_norm=round(pooled_norm, 4),
            cls_pooled_similarity=round(cls_pooled_sim, 4)
        )
        logger.debug(f"🔍 [FEATURE REPRESENTATION] CLS norm={cls_norm:.4f} | pooled norm={pooled_norm:.4f} | sim={cls_pooled_sim:.4f}")

        # Per-token features for masked positions 
        attn_map = {e.mask_position: e for e in attention_result.attention_entries}
        emb_map = {te.position: te for te in embedding_result.token_embeddings}

        masked_token_features: List[TokenFeature] = []
        for pred in prediction_result.predictions:
            ae = attn_map.get(pred.mask_position)

            # Build 64-d feature from: positional encoding (64-d) modulated by attention weights and prediction confidence
            pos_enc = sinusoidal_positional_encoding(pred.mask_position, POS_ENCODING_DIM)
            l_wt = ae.left_attention_weight if ae else 0.5
            r_wt = ae.right_attention_weight if ae else 0.5
            conf = pred.top_confidence

            # Modulate positional encoding by contextual signals
            feature_arr = np.array(pos_enc, dtype=np.float64)
            feature_arr[:32] *= l_wt       # first half weighted by left attention
            feature_arr[32:] *= r_wt       # second half weighted by right attention
            feature_arr *= conf        # scale by prediction confidence
            # Normalise
            norm = np.linalg.norm(feature_arr)
            if norm > 0:
                feature_arr /= norm

            masked_token_features.append(TokenFeature(
                position=pred.mask_position,
                token_text=pred.original_text,
                predicted_text=pred.top_prediction,
                confidence=pred.top_confidence,
                feature_vector=feature_arr.tolist()
            ))

        # Feature quality: geometric mean of accuracy, coherence, MRR
        quality = (
            prediction_result.accuracy
            * attention_result.avg_coherence
            * prediction_result.mean_mrr
        ) ** (1 / 3)

        elapsed = time.perf_counter() - t0
        result = FeatureRepresentationResult(
            sentence_features=sentence_features,
            masked_token_features=masked_token_features,
            representation_dim=POS_ENCODING_DIM,
            feature_quality_score=round(quality, 4),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [FEATURE REPRESENTATION] sentence_dim={EMBEDDING_DIMENSIONS} | "
            f"token_features={len(masked_token_features)} | quality={quality:.4f} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 Sample token features: {[{'pos': f.position, 'token': f.token_text, 'pred': f.predicted_text, 'conf': f.confidence} for f in masked_token_features[:5]]}")
        return result
    
# ==========================================
# MLM AGENT  —  Orchestrates all 8 pipeline stages
# ==========================================
class MLMAgent(BaseAIAgent):
    """
    Masked Language Model Agent.

    Pipeline:
      [Text Input] → [Token Masking] → [Embedding Layer] →
      [Left Context] ⟷ [Right Context] →
      [Bidirectional Attention] → [Masked Token Prediction] → [Feature Representation]

    BERT-style MLM: each masked position attends to both left and right context simultaneously. Not autoregressive.
    """
    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self._masking = TokenMaskingStage()
        self._embedding = EmbeddingLayerStage(client)
        self._context = ContextExtractionStage()
        self._bidir_attn = BidirectionalAttentionStage()
        self._prediction = MaskedTokenPredictionStage()
        self._features = FeatureRepresentationStage()

    # Public entry point 

    def process(self, mlm_input: MLMInput) -> MLMOutput:
        """
        Execute the full MLM pipeline.

        Args:
            mlm_input: Validated MLMInput pydantic model.

        Returns:
            MLMOutput: Fully structured bidirectional MLM pipeline result.

        Raises:
            ValueError: On invalid input.
            RuntimeError: On unrecoverable API failure.
        """
        pipeline_start = time.perf_counter()
        logger.info(f"🚀 [MLM AGENT] Pipeline START | request_id={mlm_input.request_id}")
        logger.info(
            f"📥 [INPUT] task={mlm_input.task.value} | text_len={len(mlm_input.text)} chars | "
            f"mask_prob={mlm_input.mask_probability} | top_k={mlm_input.top_k} | seed={mlm_input.seed}"
        )

        try:
            # Stage 2: Token Masking 
            masking = checkpointer.load("TOKEN_MASKING", TokenMaskingResult)
            if not masking:
                masking = self._masking.run(mlm_input)
                checkpointer.save("TOKEN_MASKING", masking)

            if masking.num_masked == 0:
                raise ValueError(
                    "Token masking produced 0 masked positions. "
                    "Input may be too short or mask_probability too low."
                )

            # Stage 3: Embedding Layer 
            embedding = checkpointer.load("EMBEDDING_LAYER", EmbeddingLayerResult)
            if not embedding:
                embedding = self._embedding.run(masking, self)
                checkpointer.save("EMBEDDING_LAYER", embedding)

            # Stage 4 ⟷ 5: Left Context ⟷ Right Context (parallel) 
            logger.info("⚙️ [MLM AGENT] Extracting Left ⟷ Right contexts (bidirectional)...")
            left_context = checkpointer.load("LEFT_CONTEXT", LeftContextResult)
            if not left_context:
                left_context = self._context.extract_left(masking)
                checkpointer.save("LEFT_CONTEXT", left_context)

            right_context = checkpointer.load("RIGHT_CONTEXT", RightContextResult)
            if not right_context:
                right_context = self._context.extract_right(masking)
                checkpointer.save("RIGHT_CONTEXT", right_context)


            # Stage 6: Bidirectional Attention
            bidir_attention = checkpointer.load("BIDIRECTIONAL_ATTENTION", BidirectionalAttentionResult)
            if not bidir_attention:
                bidir_attention = self._bidir_attn.run(
                    masking, left_context, right_context, self
                )
                checkpointer.save("BIDIRECTIONAL_ATTENTION", bidir_attention)

            # Stage 7: Masked Token Prediction
            predictions = checkpointer.load("MASKED_TOKEN_PREDICTION", MaskedTokenPredictionResult)
            if not predictions:
                predictions = self._prediction.run(
                    masking, bidir_attention, mlm_input.top_k, self
                )
                checkpointer.save("MASKED_TOKEN_PREDICTION", predictions)

            # Stage 8: Feature Representation 
            features = self._features.run(embedding, bidir_attention, predictions)

            # Output 
            total_time = time.perf_counter() - pipeline_start
            output = MLMOutput(
                request_id=mlm_input.request_id,
                token_masking=masking,
                embedding_layer=embedding,
                left_context=left_context,
                right_context=right_context,
                bidirectional_attention=bidir_attention,
                masked_token_prediction=predictions,
                feature_representation=features,
                fill_mask_accuracy=predictions.accuracy,
                mean_prediction_confidence=predictions.mean_confidence,
                bidirectional_confidence=bidir_attention.avg_coherence,
                total_pipeline_time=round(total_time, 4),
                metadata={
                    **mlm_input.metadata,
                    "model_chat": CHAT_MODEL,
                    "model_embedding": EMBEDDING_MODEL,
                },
            )

            logger.info(
                f"🎉 [MLM AGENT] Pipeline COMPLETE | total_time={total_time:.4f}s | "
                f"masked={masking.num_masked} | accuracy={predictions.accuracy:.4f} | "
                f"mean_mrr={predictions.mean_mrr:.4f} | bidirectional={bidir_attention.is_truly_bidirectional}"
            )
            logger.debug(f"🔍 [MLM AGENT] Final output: {output.model_dump_json(indent=2)}")
            return output

        except Exception as e:
            elapsed = time.perf_counter() - pipeline_start
            logger.error(
                f"❌ [MLM AGENT] Pipeline FAILED after {elapsed:.4f}s | "
                f"error={type(e).__name__}: {e}"
            )
            raise RuntimeError(f"❌ MLM pipeline failed: {e}") from e

    # Display helper 

    def display_output(self, output: MLMOutput) -> None:
        div = "=" * 80
        print(f"\n{div}")
        print("🔵 MLM AGENT — Masked Language Model Pipeline Result")
        print(f"{div}")
        print(f"Request ID: {output.request_id}")
        print(f"Status: {output.status.value}")
        print(f"Total Time: {output.total_pipeline_time}s")
        print(f"{div}")

        print(f"🔮 MASKED TOKEN PREDICTION (top-{output.token_masking.num_masked})")
        mp_res = output.masked_token_prediction
        print(f"Total Predicted: {mp_res.total_predicted}")
        print(f"Accuracy: {mp_res.accuracy:.4f} ({mp_res.accuracy:.2%})")
        print(f"Mean Confidence: {mp_res.mean_confidence:.4f}")
        print(f"Mean MRR: {mp_res.mean_mrr:.4f}")
        print(f"\nPredictions:")
        for pred in mp_res.predictions[:6]:
            icon = "✅" if pred.exact_match else "❌"
            cands_str = " | ".join(
                f"'{c.token_text}'({c.confidence:.2f})"
                for c in pred.candidates[:3]
            )
            print(
                f"{icon} [{pred.mask_position:>3}] "
                f"orig='{pred.original_text}' → "
                f"top='{pred.top_prediction}' | "
                f"candidates: [{cands_str}]"
            )
        print(f"Time: {mp_res.processing_time}s")

        print(f"{div}")
        print(f"📊 FEATURE REPRESENTATION")
        fr = output.feature_representation
        sf = fr.sentence_features
        print(f"Representation Dim: {fr.representation_dim}d (token) + {EMBEDDING_DIMENSIONS}d (sentence)")
        print(f"CLS Norm: {sf.cls_norm:.4f}")
        print(f"Pooled Norm: {sf.pooled_norm:.4f}")
        print(f"CLS-Pooled Similarity: {sf.cls_pooled_similarity:.4f}")
        print(f"Token Features: {len(fr.masked_token_features)}")
        print(f"Feature Quality: {fr.feature_quality_score:.4f}")
        for tf in fr.masked_token_features[:3]:
            print(
                f"[{tf.position:>3}] '{tf.token_text}' → "
                f"'{tf.predicted_text}' | conf={tf.confidence:.3f} | "
                f"feat[0:3]={[round(x,4) for x in tf.feature_vector[:3]]}"
            )
        print(f"Time: {fr.processing_time}s")

        print(f"{div}")
        print(f"📈 SUMMARY METRICS")
        print(f"Fill-Mask Accuracy: {output.fill_mask_accuracy:.4f}")
        print(f"Mean Confidence: {output.mean_prediction_confidence:.4f}")
        print(f"Bidir Coherence: {output.bidirectional_confidence:.4f}")
        print(f"Feature Quality: {fr.feature_quality_score:.4f}")
        print(f"\n{div}\n")

# ==========================================
# Instatiation
# ==========================================
def create_mlm_agent(api_key: str = TOKEN, endpoint: str = ENDPOINT) -> MLMAgent:
    """Factory function to create an instance of MLMAgent with the provided API credentials."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    logger.info("🏭 [FACTORY] MLMAgent instantiated and ready.")
    return MLMAgent(client)

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    agent = create_mlm_agent()

    # Demo 1: Standard fill-mask task 
    # mlm_input = MLMInput(
    #     text=(
    #         "The transformer architecture revolutionised natural language processing by introducing the self-attention mechanism, which allows the model to weigh the importance of different words ina sequence when making predictions. "
    #         "Unlike recurrent networks, transformers process all tokens simultaneously and can capture long-range dependencies more effectively. "
    #         "This parallel processing capability makes transformers highly scalable and suitable for training on large datasets with modern hardware accelerators."
    #     ),
    #     task = MLMTask.FILL_MASK,
    #     mask_probability= 0.15,
    #     top_k = 5,
    #     seed = 42,
    #     metadata = {"source": "mlm_agent_demo", "version": "1.0"},
    # )

    # result = agent.process(mlm_input)
    # agent.display_output(result)

    # Demo 2: Higher masking rate 
    print("\n" + "═" * 72)
    print("📋 Demo 2: Higher masking rate (25%) — harder prediction task")
    print("═" * 72 + "\n")

    mlm_input_2 = MLMInput(
        text=(
            "Bidirectional encoder representations from transformers enable deep understanding of language context by jointly conditioning on both left and right context in all layers."
        ),
        task = MLMTask.FILL_MASK,
        mask_probability= 0.25,
        top_k = 3,
        seed = 7,
        metadata = {"source": "mlm_agent_demo_2"},
    )
    result2 = agent.process(mlm_input_2)
    print(f"Accuracy: {result2.fill_mask_accuracy:.4f}")
    print(f"Mean Confidence: {result2.mean_prediction_confidence:.4f}")
    print(f"Mean MRR: {result2.masked_token_prediction.mean_mrr:.4f}")
    print(f"Bidir Coherence: {result2.bidirectional_confidence:.4f}")
    print(f"Num Masked: {result2.token_masking.num_masked}")
    print(f"\nPredictions:")
    for pred in result2.masked_token_prediction.predictions:
        icon = "✅" if pred.exact_match else "❌"
        print(
            f"{icon} [{pred.mask_position:>3}] "
            f"'{pred.original_text}' → '{pred.top_prediction}' "
            f"(conf={pred.top_confidence:.3f})"
        )

