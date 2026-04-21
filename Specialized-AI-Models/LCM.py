"""
LCM AGENT — Large Concept Model

Pipeline:
Input → Sentence Segmentation → SONAR Embedding → Diffusion → [Advanced Patterning ⟷ Hidden Process] → Quantization → Output

Model: gpt-4.1 + text-embedding-3-small (SONAR-style pooling)
Standard: Production-Grade | Pydantic v2 | ABC | Retry Logic

Architectural Note:
Meta's SONAR uses LASER2 sentence encoders for language-agnostic concept embeddings. This is replicated via OpenAI per-sentence embeddings with mean-pooled concept vectors, preserving the architecture without local model dependencies.
"""

from abc import ABC, abstractmethod
from typing import Any
import time
from datetime import datetime, timezone
from openai import OpenAI, RateLimitError, APITimeoutError, APIError
import nltk
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid
import numpy as np
import json

import os
from dotenv import load_dotenv
load_dotenv()

from logging_setup import get_logger
logger = get_logger(__name__, log_file="lcm.log")

# ==========================================
# Variable Configuration
# ==========================================
TOKEN = os.environ['GITHUB_TOKEN']
ENDPOINT = os.environ['GITHUB_ENDPOINT']
CHAT_MODEL = os.environ['GITHUB_MODEL_NAME']
EMBEDDING_MODEL = os.environ['GITHUB_EMBED_NAME']

# Diffusion
DIFFUSION_STEPS = 4 # refinement iterations
DIFFUSION_NOISE_SCALE = 0.03 # Gaussian noise magnitude per step

# Quantization
QUANTIZATION_BITS = 8 # scalar quantization depth => 256 levels
CODEBOOK_SIZE = 256 # 2^8

# Clustering (Hidden Process)
SIMILARITY_THRESHOLD = 0.8 # cosine sim threshold for concept clustering

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

ADVANCED_PATTERNING_PROMPT = """
You are a concept-modelling engine performing Advanced Patterning.

Analyse the following {len_sentences} sentences at abstraction depth {concept_depth_value}/5
(where 1=surface patterns, 5=deep philosophical/structural patterns).

SENTENCES:
{sentences_text}

Extract exactly {number_of_conceptual_patterns} conceptual patterns.

Respond ONLY with valid JSON in this exact structure:
{{
  "patterns": [
    {{
      "pattern_id": "P001",
      "label": "<short_pattern_name>",
      "description": "<one_sentence_description>",
      "confidence": <float 0.0–1.0>
    }}
  ],
  "dominant_theme": "<single dominant concept theme across all sentences>"
}}
"""

# ==========================================
# Ensure NLTK data is present
# ==========================================
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ==========================================
# Pydantic Models
# ==========================================
class LCMInput(BaseModel):
    """Stage 1: Validated input schema for the LCM agent."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the request")
    text: str = Field(...,min_length=10, max_length=64000, description="Raw input text to be processed by the LCM agent")
    language: str = Field(default="en", description="ISO 639-1 language code for the input text")
    concept_depth: int = Field(default=3, ge=1, le=5, description="Depth of concept extraction (1=surface, 3=balanced, 5=deep)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata for contextual processing (e.g., domain, user profile)")

    @field_validator('text')
    @classmethod
    def validate_text(cls, v):
        stripped = v.strip()    
        if not stripped:
            raise ValueError("Input text cannot be empty or whitespace.")
        return stripped
    
    @model_validator(mode='after')
    def stamp_metadata(self) -> "LCMInput":
        """Automatically add processing timestamp to metadata."""
        self.metadata['created_at'] = datetime.now(timezone.utc).isoformat()
        self.metadata['language'] = self.language
        return self
    
class Sentence(BaseModel):
    """Represents a single segmented sentence with its index."""
    text: str
    index: int
    char_length: int

class SegmentationResult(BaseModel):
    """Stage 2: Output of sentence segmentation stage."""

    stage: str = 'SENTENCE_SEGMENTATION'
    sentences: list[Sentence]
    sentence_count: int
    processing_time: float

class ConceptVector(BaseModel):
    """Represents a single SONAR-style concept vector derived from sentence embeddings."""
    vector: list[float]
    sentence_index: int

class SonarEmbeddingResult(BaseModel):
    """Stage 3: Output of SONAR embedding stage."""

    stage: str = 'SONAR_EMBEDDING'
    model: str
    concept_vectors: list[ConceptVector]
    pooled_concept_vector: list[float]
    dimensions: int
    processing_time: float

class DiffusionStep(BaseModel):
    """Represents the output of a single diffusion refinement step."""
    step: int
    noise_scale: float
    delta_norm: float # L2 norm of the update applied

class DiffusionResult(BaseModel):
    """Stage 4: Output of the diffusion refinement stage."""

    stage: str = 'DIFFUSION'
    refined_vector: list[float] # output concept vector after diffusion
    steps: list[DiffusionStep]
    total_drift: float # cumulative L2 change from original
    processing_time: float

class ConceptPattern(BaseModel):
    """Represents a single discovered concept pattern from the hidden process."""
    pattern_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the concept pattern")
    label: str
    description: str
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score for the pattern (0.0 to 1.0)")

class AdvancedPatterningResult(BaseModel):
    """Stage 5: Advanced Patterning output (GPT-4.1 structural analysis)."""

    stage: str = 'ADVANCED_PATTERNING'
    patterns: list[ConceptPattern]
    dominant_theme: str
    processing_time: float

class ConceptCluster(BaseModel):
    """Represents a cluster of conceptually related sentence indices identified in the hidden process."""
    
    cluster_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for the concept cluster")
    sentence_indices: list[int] # indices of concept vectors in this cluster
    centroid_vector: list[float]
    cohesion_score: float = Field(ge=0.0, le=1.0, description="Cohesion score for the cluster (0.0 to 1.0)")

class HiddenProcessResult(BaseModel):
    """Stage 6: Output of the hidden process stage, including concept clusters."""

    stage: str = 'HIDDEN_PROCESS'
    clusters: list[ConceptCluster]
    cluster_count: int
    cross_cluster_insight: str
    processing_time: float

class QuantizedCode(BaseModel):
    """Quantized scalar code for one concept vector dimension bucket."""

    bucket_id: int
    code: int
    range_min: float
    range_max: float

class QuantizationResult(BaseModel):
    """Stage 7: Output of the quantization stage, including quantized codes and codebook."""

    stage: str = 'QUANTIZATION'
    bits: int
    codebook_size: int
    codes: list[int] # full quantized representation
    compression_ratio: float # original_floats / code_bytes
    processing_time: float

class LCMOutput(BaseModel):
    """Stage 8 — Final structured output of the full LCM pipeline."""

    request_id: str
    stage: str = 'OUTPUT'
    status: str = Field(default="SUCCESS", description="Overall status of the LCM processing")

    # payloads
    segmentation: SegmentationResult
    sonar_embedding: SonarEmbeddingResult
    diffusion: DiffusionResult
    advanced_patterning: AdvancedPatterningResult
    hidden_process: HiddenProcessResult
    quantization: QuantizationResult

    # summary
    concept_summary: str
    total_pipeline_time: float
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the processing (e.g., model versions, processing environment)")

    @model_validator(mode='after')
    def populate_metadata(self) -> "LCMOutput":
        """Automatically populate metadata with processing environment details."""
        
        self.metadata['completed_at'] = datetime.now(timezone.utc).isoformat()
        self.metadata['sentence_count'] = self.segmentation.sentence_count
        self.metadata['cluster_count'] = self.hidden_process.cluster_count
        self.metadata['pattern_count'] = len(self.advanced_patterning.patterns)
        return self

# ==========================================
# ABSTRACT BASE  (shared across all 8 agents)
# ==========================================

class BaseAIAgent(ABC):
    """Abstract base class defining the shared contract for all 8 AI agents."""

    def __init__(self, client: OpenAI | None):
        if client is not None:
            self.client = client
        else:
            self.client = OpenAI(
                base_url=ENDPOINT,
                api_key=TOKEN,
            )
        self.logger = get_logger(__name__, log_file="lcm.log")
    
    @abstractmethod
    def process(self, input_data: Any) -> Any:
        """Core execution pipeline to be implemented by each specialized agent."""
        ...

    def _retry_api_call(self, fn, *args, **kwargs):
        """
        Exponential-backoff retry wrapper for any OpenAI API call.
        Handles: RateLimitError, APITimeoutError, APIError.
        """
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return fn(*args, **kwargs)
            except RateLimitError as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                self.logger.warning(
                    f"🔄 Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait}s... | {e}"
                )
                time.sleep(wait)
            except APITimeoutError as e:
                wait = RETRY_BACKOFF_BASE ** attempt
                self.logger.warning(
                    f"⏱️  Timeout (attempt {attempt}/{MAX_RETRIES}). "
                    f"Retrying in {wait}s... | {e}"
                )
                time.sleep(wait)
            except APIError as e:
                self.logger.error(f"API error on attempt {attempt}: {e}")
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(RETRY_BACKOFF_BASE ** attempt)

        raise RuntimeError(f"All {MAX_RETRIES} API retry attempts exhausted.")
    
# ==========================================
# Helper Functions
# ==========================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1-D vectors."""
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def mean_pool(vectors: list[np.ndarray]) -> np.ndarray:
    """Mean-pool a list of equal-dim vectors → single concept vector."""
    return np.mean(np.stack(vectors, axis=0), axis=0)

# ==========================================
# Pipeline Stages
# ==========================================
class SentenceSegmentationStage:
    """
    Stage 2: SENTENCE SEGMENTATION
    Uses NLTK punkt tokenizer to split input into atomic conceptual units.
    Each sentence becomes an independent concept carrier for SONAR embedding.
    """

    def run(self, text: str) -> SegmentationResult:
        logger.info("⚙️ [SEGMENTATION] Splitting text into concept units...")
        t0 = time.perf_counter()

        raw_sentences = nltk.sent_tokenize(text)
        sentences = [
            Sentence(index=i, text=s.strip(), char_length=len(s.strip()))
            for i, s in enumerate(raw_sentences)
            if s.strip()
        ]
        elapsed = time.perf_counter() - t0
        result = SegmentationResult(
            sentences=sentences,
            sentence_count=len(sentences),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [SEGMENTATION] {result.sentence_count} sentences extracted | "
            f"time={elapsed:.4f}s"
        )
        return result

class SonarEmbeddingStage:
    """
    Stage 3: SONAR EMBEDDING
    Encodes each sentence independently into concept space, then mean-pools all vectors into a single pooled concept representation.

    SONAR Fidelity Notes:
    - Original SONAR: LASER2 encoder, truly language-agnostic via shared multilingual sentence space.
    - My implementation: Per-sentence OpenAI embeddings (text-embedding-3-small) + mean pooling — same architecture, API-native execution.
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, sentences: list[Sentence], retry_fn) -> SonarEmbeddingResult:
        logger.info(f"⚙️ [SONAR EMBEDDING] Encoding {len(sentences)} sentences into concept space...")
        t0 = time.perf_counter()
        concept_vectors: list[ConceptVector] = []
        np_vectors: list[np.ndarray] = []

        for sentence in sentences:
            response = retry_fn(
                self._client.embeddings.create,
                model=EMBEDDING_MODEL,
                input=sentence.text
            )
            vector = response.data[0].embedding
            concept_vectors.append(
                ConceptVector(vector=vector, sentence_index=sentence.index)
            )
            np_vectors.append(np.array(vector, dtype=np.float32))
            logger.debug(
                f"🔍 [SONAR] Encoded sentence {sentence.index} | "
                f"({sentence.char_length} chars)"
            )
        
        # Mean-pool -> single language-agnostic concept vector
        pooled = mean_pool(np_vectors).tolist()
        elapsed = time.perf_counter() - t0
        result = SonarEmbeddingResult(
            model=EMBEDDING_MODEL,
            concept_vectors=concept_vectors,
            pooled_concept_vector=pooled,
            dimensions=len(pooled),
            processing_time=round(elapsed, 4)
        )
        logger.debug(
            f"✅ [SONAR EMBEDDING] {len(concept_vectors)} vectors encoded | "
            f"pooled dim={result.dimensions} | time={elapsed:.4f}s"
        )
        return result

class DiffusionStage:
    """
    Stage 4: DIFFUSION
    Applies iterative concept refinement over the pooled embedding vector.

    Process (DDPM-inspired, adapted for concept space):
    1. Add controlled Gaussian noise to simulate forward diffusion.
    2. Apply a reverse denoising pass via signed normalisation.
    3. Repeat for DIFFUSION_STEPS iterations.

    This models conceptual 'sharpening' — starting from a noisy latent concept and converging toward a semantically stable representation.
    """

    def run(self, pooled_vector: list[float]) -> DiffusionResult:
        logger.info(f"⚙️ [DIFFUSION] Starting {DIFFUSION_STEPS}-step concept refinement...")
        t0     = time.perf_counter()
        vector    = np.array(pooled_vector, dtype=np.float64)
        origin = vector.copy()
        steps: list[DiffusionStep] = []

        rng = np.random.default_rng(seed=42) # deterministic noise for reproducibility

        for step in range(1, DIFFUSION_STEPS + 1):
            noise_scale = DIFFUSION_NOISE_SCALE / step   # decay noise each step

            # Forward: inject noise
            noise = rng.normal(0, noise_scale, vector.shape)
            noisy = vector + noise

            # Reverse: denoise via L2-normalised projection
            denoised = noisy / (np.linalg.norm(noisy) + 1e-8)

            # Residual update (concept refinement delta)
            delta     = denoised - vector
            delta_norm = float(np.linalg.norm(delta))
            vector       = vector + 0.5 * delta       # soft update

            steps.append(DiffusionStep(
                step       = step,
                noise_scale= round(noise_scale, 6),
                delta_norm = round(delta_norm, 6),
            ))
            logger.debug(
                f"🌀 [DIFFUSION] Step {step}/{DIFFUSION_STEPS} | "
                f"noise={noise_scale:.6f} | delta_norm={delta_norm:.6f}"
            )

        total_drift = float(np.linalg.norm(vector - origin))
        elapsed     = time.perf_counter() - t0

        result = DiffusionResult(
            steps          = steps,
            refined_vector = vector.tolist(),
            total_drift    = round(total_drift, 6),
            processing_time= round(elapsed, 4),
        )
        logger.info(f"✅ [DIFFUSION] Complete | total_drift={total_drift:.6f} | time={elapsed:.4f}s")
        return result

class AdvancedPatterningStage:
    """
    Stage 5: ADVANCED PATTERNING
    Uses GPT-4.1 to perform structural analysis over segmented sentences,
    extracting abstract conceptual patterns — themes, relationships, and
    emergent structures not visible at the surface level.
    """
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, sentences: list[Sentence], concept_depth: int, retry_fn) -> AdvancedPatterningResult:
        logger.info(
            f"⚙️ [ADVANCED PATTERNING] Extracting abstract patterns "
            f"at depth={concept_depth}..."
        )
        t0 = time.perf_counter()

        sentences_text = "\n".join(
            f"[{s.index}] {s.text}" for s in sentences
        )
        prompt = ADVANCED_PATTERNING_PROMPT.format(
            len_sentences=len(sentences),
            concept_depth_value=concept_depth,
            sentences_text=sentences_text,
            number_of_conceptual_patterns=min(concept_depth + 2, 6)
        )
        response = retry_fn(
            self._client.chat.completions.create,
            model=CHAT_MODEL,
            messages   = [
                {
                    "role": "system",
                    "content": (
                        "You are a precision concept-extraction engine. "
                        "Output only valid JSON. No preamble, no markdown fences."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature = 0.2,
            max_tokens  = 1024,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content or '{}'
        data = json.loads(raw)
        patterns = [ConceptPattern(**p) for p in data.get("patterns", [])]
        dominant = data.get("dominant_theme", "Unknown")
        elapsed = time.perf_counter() - t0
        result = AdvancedPatterningResult(
            patterns=patterns,
            dominant_theme=dominant,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [ADVANCED PATTERNING] {len(patterns)} patterns | "
            f"dominant='{dominant}' | time={elapsed:.4f}s"
        )
        return result

class HiddenProcessStage:
    """
    Stage 6: HIDDEN PROCESS
    Performs latent concept clustering over per-sentence embeddings using
    cosine similarity. Conceptually related sentences are grouped into clusters,
    then GPT-4.1 generates a cross-cluster inference — the 'hidden' emergent
    concept that links all clusters.

    This is the LCM's core differentiator: operating on latent concept space
    rather than surface token space.
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, concept_vectors: list[ConceptVector], sentences: list[Sentence], retry_fn) -> HiddenProcessResult:
        logger.info(
            f"⚙️ [HIDDEN PROCESS] Clustering {len(concept_vectors)} "
            f"concept vectors..."
        )
        t0 = time.perf_counter()

        np_vecs = [
            np.array(cv.vector, dtype=np.float32)
            for cv in concept_vectors
        ]

        # greedy cosine similarity clustering
        assigned   = [False] * len(np_vecs)
        clusters: list[ConceptCluster] = []
        cluster_id = 0

        for i, vec_i in enumerate(np_vecs):
            if assigned[i]:
                continue
            group = [i]
            assigned[i] = True
            for j, vec_j in enumerate(np_vecs):
                if not assigned[j] and i != j:
                    sim = cosine_similarity(vec_i, vec_j)
                    if sim >= SIMILARITY_THRESHOLD:
                        group.append(j)
                        assigned[j] = True

            group_vecs = [np_vecs[g] for g in group]
            centroid = mean_pool(group_vecs)
            cohesion = float(np.mean([
                cosine_similarity(centroid, v) for v in group_vecs
            ]))

            clusters.append(ConceptCluster(
                cluster_id=str(cluster_id),
                sentence_indices=group,
                centroid_vector=centroid.tolist(),
                cohesion_score=round(cohesion, 4),
            ))
            cluster_id += 1
            logger.debug(
                f"🔗 [HIDDEN PROCESS] Cluster {cluster_id} | "
                f"sentences={group} | cohesion={cohesion:.4f}"
            )
        
        # cross-cluster GPT inference
        cluster_desc = "\n".join([
            f"Cluster {c.cluster_id}: sentences "
            f"{[sentences[i].text[:60] for i in c.sentence_indices]}"
            for c in clusters
        ])
        insight_response = retry_fn(
            self._client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a latent concept analyst. "
                        "Identify the single  hidden emergent concept that bridges all clusters. "
                        "Respond in one concise sentence only."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Given these concept clusters:\n{cluster_desc}\n\n"
                        "What is the hidden emergent concept connecting them all?"
                    ),
                },
            ],
            temperature=0.3,
            max_tokens =150,
        )
        insight = insight_response.choices[0].message.content or ""
        elapsed = time.perf_counter() - t0

        result = HiddenProcessResult(
            clusters=clusters,
            cluster_count=len(clusters),
            cross_cluster_insight=insight.strip(),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [HIDDEN PROCESS] {len(clusters)} clusters | "
            f"time={elapsed:.4f}s"
        )
        return result

class QuantizationStage:
    """
    Stage 7: QUANTIZATION
    Compresses the refined diffusion vector into discrete scalar codes.

    Method — Uniform Scalar Quantization:
    1. Compute global min/max of the refined vector.
    2. Divide the range into 2^QUANTIZATION_BITS equal bins.
    3. Map each float value → integer code (0 to 255 for 8-bit).

    This produces a compact discrete concept representation — analogous to
    how LCMs compress continuous latent concepts into discrete codebooks
    for efficient storage and cross-lingual transfer.
    """

    def run(self, refined_vector: list[float]) -> QuantizationResult:
        logger.info(
            f"⚙️ [QUANTIZATION] Quantizing {len(refined_vector)}-dim "
            f"vector to {QUANTIZATION_BITS}-bit codes..."
        )
        t0  = time.perf_counter()
        vector = np.array(refined_vector, dtype=np.float64)
        v_min, v_max = float(vector.min()), float(vector.max())
        v_range      = v_max - v_min + 1e-10   # avoid div-by-zero

        # Uniform quantization
        codes = np.floor(
            (vector - v_min) / v_range * (CODEBOOK_SIZE - 1)
        ).astype(np.int32).clip(0, CODEBOOK_SIZE - 1)

        # Compression ratio: original float32 bytes vs int8 bytes
        original_bytes = len(vector) * 4 # float32 = 4 bytes
        compressed_bytes = len(codes) * 1 # int8   = 1 byte
        compression_ratio = original_bytes / compressed_bytes

        elapsed = time.perf_counter() - t0
        result  = QuantizationResult(
            bits=QUANTIZATION_BITS,
            codebook_size=CODEBOOK_SIZE,
            codes=codes.tolist(),
            compression_ratio=round(compression_ratio, 2),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [QUANTIZATION] {len(codes)} codes | "
            f"range=[{v_min:.4f}, {v_max:.4f}] | "
            f"compression={compression_ratio:.1f}x | "
            f"time={elapsed:.4f}s"
        )
        return result

# ==========================================
# LCM Agent Implementation
# ==========================================
class LCMAgent(BaseAIAgent):
    """LCM Agent orchestrating the full pipeline from input to final output."""

    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self._segmenter = SentenceSegmentationStage()
        self._sonar = SonarEmbeddingStage(client)
        self._diffusion = DiffusionStage()
        self._patterning = AdvancedPatterningStage(client)
        self._hidden = HiddenProcessStage(client)
        self._quantization = QuantizationStage()

    def process(self, lcm_input: LCMInput) -> LCMOutput:
        """
        Execute the full LCM pipeline.

        Args:
            lcm_input: Validated LCMInput pydantic model.

        Returns:
            LCMOutput: Fully structured multi-stage pipeline result.

        Raises:
            ValueError: On invalid input.
            RuntimeError: On unrecoverable API failure.
        """
        pipeline_start = time.perf_counter()
        self.logger.info(
            f"🚀 [LCM AGENT] Pipeline START | "
            f"request_id={lcm_input.request_id} | "
            f"language={lcm_input.language} | "
            f"depth={lcm_input.concept_depth}"
        )
        self.logger.info(
            f"📥 [INPUT] {len(lcm_input.text)} chars received"
        )

        try:
            # Stage 2: Sentence Segmentation
            segmentation = self._segmenter.run(lcm_input.text)
            if segmentation.sentence_count == 0:
                raise ValueError("Segmentation produced zero sentences.")
            
            # Stage 3: SONAR Embedding
            sonar = self._sonar.run(segmentation.sentences, self._retry_api_call)

            # Stage 4: Diffusion
            diffusion = self._diffusion.run(sonar.pooled_concept_vector)

            # Stage 5: Advanced Patterning
            # Stage 6: Hidden Process
            # both run in parallel conceptually; share same sentences + vector data
            self.logger.info("⚙️ [LCM AGENT] Running Advanced Patterning <-> Hidden Process (interleaved)...")
            patterning = self._patterning.run(
                segmentation.sentences,
                lcm_input.concept_depth,
                self._retry_api_call,
            )
            hidden = self._hidden.run(
                sonar.concept_vectors,
                segmentation.sentences,
                self._retry_api_call,
            )

            # Stage 7: Quantization
            quantization = self._quantization.run(diffusion.refined_vector)

            # Stage 8: Final output stage
            concept_summary = self._synthesise_summary(patterning, hidden, lcm_input)
            total_time = time.perf_counter() - pipeline_start
            output = LCMOutput(
                request_id=lcm_input.request_id,
                segmentation=segmentation,
                sonar_embedding=sonar,
                diffusion=diffusion,
                advanced_patterning=patterning,
                hidden_process=hidden,
                quantization=quantization,
                concept_summary=concept_summary,
                total_pipeline_time=round(total_time, 4),
                metadata={
                    **lcm_input.metadata,
                    "model_chat": CHAT_MODEL,
                    "model_embedding": EMBEDDING_MODEL,
                }
            )
            self.logger.info(
                f"🎉 [LCM AGENT] Pipeline COMPLETE | "
                f"total_time={total_time:.4f}s | "
                f"sentences={segmentation.sentence_count} | "
                f"clusters={hidden.cluster_count} | "
                f"patterns={len(patterning.patterns)}"
            )
            return output
        except Exception as e:
            elapsed = time.perf_counter() - pipeline_start
            self.logger.error(
                f"❌ [LCM AGENT] Pipeline FAILED after {elapsed:.4f}s | "
                f"error={type(e).__name__}: {e}"
            )
            raise

    # helper functions
    def _synthesise_summary(self, patterning: AdvancedPatterningResult, hidden: HiddenProcessResult, lcm_input: LCMInput) -> str:
        """Generate a final concept-level summary using all stage outputs."""
        pattern_labels = ", ".join(p.label for p in patterning.patterns)
        prompt = f"""
        Dominant theme: {patterning.dominant_theme}
        Identified patterns: {pattern_labels}
        Hidden emergent concept: {hidden.cross_cluster_insight}
        Write a single-paragraph concept-level summary of the original text at abstraction depth {lcm_input.concept_depth}/5. 
        Focus on WHAT the text means conceptually, not what it says literally.
        """
        response = self._retry_api_call(
            self.client.chat.completions.create,
            model    = CHAT_MODEL,
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a concept synthesis engine. Produce concise, insightful conceptual summaries."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens =300,
        )
        return (response.choices[0].message.content or "").strip()
    
    def display_output(self, output: LCMOutput) -> None:
        divider = "=" * 100
        print(f"\n{divider}")
        print("  🟢 LCM AGENT — Large Concept Model Pipeline Result")
        print(f"{divider}")

        # print(f"  Request ID        : {output.request_id}")
        # print(f"  Status            : {output.status.value}")
        # print(f"  Total Time        : {output.total_pipeline_time}s")
        # print(f"{divider}")
        # print(f"  ✂️  SENTENCE SEGMENTATION")
        # print(f"     Sentences       : {output.segmentation.sentence_count}")
        # for s in output.segmentation.sentences[:3]:
        #     print(f"     [{s.index}] {s.text[:80]}...")
        # print(f"     Time            : {output.segmentation.processing_time}s")
        # print(f"{divider}")
        # print(f"  🧠 SONAR EMBEDDING")
        # print(f"     Vectors Encoded : {len(output.sonar_embedding.concept_vectors)}")
        # print(f"     Dimensions      : {output.sonar_embedding.dimensions}")
        # print(f"     Pooled[0:3]     : {output.sonar_embedding.pooled_concept_vector[:3]}")
        # print(f"     Time            : {output.sonar_embedding.processing_time}s")
        # print(f"{divider}")
        # print(f"  🌀 DIFFUSION")
        # print(f"     Steps           : {len(output.diffusion.steps)}")
        # print(f"     Total Drift     : {output.diffusion.total_drift}")
        # print(f"     Time            : {output.diffusion.processing_time}s")
        # print(f"{divider}")
        # print(f"  🔬 ADVANCED PATTERNING")
        # print(f"     Dominant Theme  : {output.advanced_patterning.dominant_theme}")
        # for p in output.advanced_patterning.patterns:
        #     print(f"     [{p.pattern_id}] {p.label} (conf={p.confidence:.2f})")
        #     print(f"          → {p.description}")
        # print(f"     Time            : {output.advanced_patterning.processing_time}s")
        # print(f"{divider}")
        # print(f"  🔗 HIDDEN PROCESS")
        # print(f"     Clusters        : {output.hidden_process.cluster_count}")
        # for c in output.hidden_process.clusters:
        #     print(
        #         f"     Cluster {c.cluster_id}: sentences {c.sentence_indices} "
        #         f"| cohesion={c.cohesion_score}"
        #     )
        # print(f"     Emergent Insight: {output.hidden_process.cross_cluster_insight}")
        # print(f"     Time            : {output.hidden_process.processing_time}s")
        # print(f"{divider}")
        # print(f"  📦 QUANTIZATION")
        # print(f"     Bits            : {output.quantization.bits}")
        # print(f"     Codebook Size   : {output.quantization.codebook_size}")
        # print(f"     Compression     : {output.quantization.compression_ratio}x")
        # print(f"     Codes[0:8]      : {output.quantization.codes[:8]}")
        # print(f"     Time            : {output.quantization.processing_time}s")

        print(f"{divider}")
        print(f"  📤 CONCEPT SUMMARY\n")
        print(f"  {output.concept_summary}")
        print(f"\n{divider}\n")

# ==========================================
# Instatiation
# ==========================================
def create_lcm_agent(api_key, endpoint) -> LCMAgent:
    """Factory function to create an instance of LCMAgent with the provided API credentials."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    logger.info("🏭 [FACTORY] LLMAgent instantiated and ready.")
    return LCMAgent(client)

# ==========================================
# Entry Point
# ==========================================
if __name__ == "__main__":
    # ── 1. Create agent via factory ────────────────────────────────────────
    agent = create_lcm_agent(TOKEN, ENDPOINT)

    input_text = """
    The transformer architecture fundamentally changed how we model sequences.
    Attention mechanisms allow models to relate any two positions in a sequence, regardless of distance.
    This broke the bottleneck of recurrent networks which processed tokens one at a time.
    Scaling transformers revealed emergent capabilities: reasoning, coding, and even scientific discovery became possible.
    Today, large language models built on transformers are reshaping industries from healthcare to education and creative arts.
    The question is no longer whether AI can understand language — it is whether we can understand what AI truly knows.
    """

    # ── 2. Build validated input ───────────────────────────────────────────
    lcm_input = LCMInput(
        text=input_text,
        language     = "en",
        concept_depth= 4,
        metadata     = {"source": "lcm_agent_demo", "version": "1.0.0"},
    )

    # ── 3. Execute pipeline ────────────────────────────────────────────────
    result = agent.process(lcm_input)

    # ── 4. Display structured output ───────────────────────────────────────
    agent.display_output(result)

    # ── 5. Export as JSON (excluding large embedding vectors for brevity) ──
    # print("📦 Pydantic JSON (embeddings excluded for brevity):")
    # print(result.model_dump_json(
    #     indent=2,
    #     exclude={
    #         "sonar_embedding": {"concept_vectors": True, "pooled_concept_vector": True},
    #         "diffusion"      : {"refined_vector": True},
    #         "quantization"   : {"codes": True},
    #     }
    # ))
    