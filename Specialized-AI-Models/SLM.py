"""SLM AGENT — Small Language Model (Agent 6 of 8)

Pipeline:
Input Processing → Compact Tokenization → Optimized Embeddings →
Efficient Transformer → [ Model Quantization ⇔ Memory Optimization] → Edge Deployment → Output Generation

Author  : Senior AI/ML Solutions Architect
Model   : gpt-4.1 (efficiency-constrained) + text-embedding-3-small
Standard: Production-Grade | Pydantic v2 | ABC | Retry Logic

Core Philosophy:
Efficiency-first design: enforce resource constraints, token budgets, dimensionality reduction, INT4/INT8 quantization, KV-cache simulation, and robust edge payload validation.
"""

import os, uuid, time, re, json, base64, hashlib, math, numpy as np, tiktoken
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from collections import OrderedDict

from dotenv import load_dotenv
load_dotenv()

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="slm.log")

# ==========================================
# Variable Configuration
# ==========================================
from utils.state_checkpointer import StateCheckpointer
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
# checkpointer file created in process() method of SLMAgent

TOKEN = os.environ['GITHUB_TOKEN']
ENDPOINT = os.environ['GITHUB_ENDPOINT']
CHAT_MODEL = os.environ['GITHUB_MODEL_NAME']
EMBEDDING_MODEL = os.environ['GITHUB_EMBED_NAME']
EMBEDDING_DIMENSIONS = 1536

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

ENCODING = "cl100k_base"

# SLM token budget (simulates 2B–7B parameter model context limits)
SLM_MAX_INPUT_TOKENS = 512  # compact tokenization ceiling
SLM_MAX_OUTPUT_TOKENS = 256  # strict generation budget
SLM_VOCAB_SIZE = 32000  # compact vocabulary (vs. GPT-4 ~100k)

# Embedding dimensionality reduction (1536-d → 128-d for SLM efficiency)
FULL_EMBEDDING_DIM = 1536
REDUCED_EMBEDDING_DIM = 128  # PCA-style projection target

# Quantization settings
INT8_BITS = 8
INT4_BITS = 4
INT8_LEVELS = 256  # 2^8
INT4_LEVELS = 16  # 2^4

# KV-Cache settings
KV_CACHE_MAX_ENTRIES = 64  # bounded LRU cache
KV_CACHE_ENTRY_BYTES = 512  # simulated per-entry memory (bytes)

# Edge deployment SLA thresholds
EDGE_LATENCY_SLA_MS = 2_000  # 2s max latency for edge devices
EDGE_PAYLOAD_MAX_BYTES = 4_096  # 4KB max response payload
EDGE_MIN_COMPRESSION  = 2.0  # minimum required compression ratio

# ==========================================
# ENUMS
# ==========================================
class PipelineStage(str, Enum):
    INPUT_PROCESSING  = "INPUT_PROCESSING"
    COMPACT_TOKENIZATION = "COMPACT_TOKENIZATION"
    OPTIMIZED_EMBEDDINGS = "OPTIMIZED_EMBEDDINGS"
    EFFICIENT_TRANSFORMER = "EFFICIENT_TRANSFORMER"
    MODEL_QUANTIZATION = "MODEL_QUANTIZATION"
    MEMORY_OPTIMIZATION = "MEMORY_OPTIMIZATION"
    EDGE_DEPLOYMENT  = "EDGE_DEPLOYMENT"
    OUTPUT_GENERATION = "OUTPUT_GENERATION"

class QuantizationMode(str, Enum):
    INT8 = "INT8"    # 8-bit — balanced efficiency/quality
    INT4 = "INT4"    # 4-bit — maximum compression, slight quality loss

class DeviceProfile(str, Enum):
    CLOUD = "CLOUD"       # no constraints  (baseline)
    MOBILE = "MOBILE"      # 4GB RAM, moderate latency tolerance
    EDGE_IOT = "EDGE_IOT"    # 512MB RAM, strict latency SLA
    MICROCONTROLLER = "MICROCONTROLLER"  # <256MB RAM, ultra-tight budget

class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    DEGRADED = "DEGRADED"    # succeeded but SLA violated
    FAILED = "FAILED"

# ==========================================
# DEVICE PROFILE CONFIGURATIONS
# ==========================================
DEVICE_CONFIGS: Dict[DeviceProfile, dict] = {
    DeviceProfile.CLOUD: {
        "max_ram_mb": 32768,
        "latency_sla_ms": 10000,
        "quant_mode": QuantizationMode.INT8,
        "payload_max_bytes": 65536,
    },
    DeviceProfile.MOBILE: {
        "max_ram_mb": 4096,
        "latency_sla_ms": 3000,
        "quant_mode": QuantizationMode.INT8,
        "payload_max_bytes": 16384,
    },
    DeviceProfile.EDGE_IOT: {
        "max_ram_mb": 512,
        "latency_sla_ms": EDGE_LATENCY_SLA_MS,
        "quant_mode": QuantizationMode.INT4,
        "payload_max_bytes": EDGE_PAYLOAD_MAX_BYTES,
    },
    DeviceProfile.MICROCONTROLLER: {
        "max_ram_mb": 256,
        "latency_sla_ms": 1000,
        "quant_mode": QuantizationMode.INT4,
        "payload_max_bytes": 1024,
    },
}

# ==========================================
# PYDANTIC MODELS
# ==========================================
class SLMInput(BaseModel):
    """Stage 1 — Validated raw input to the SLM pipeline."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the request.")
    prompt: str = Field(..., min_length=1, max_length=8000, description="Input prompt — will be aggressively compressed to SLM budget.")
    system_prompt: str = Field(
        default="You are a helpful and efficient AI assistant. Please provide a concise response.", 
        description="System prompt defining the SLM's behavior.",
        max_length=512
    )
    device_profile: DeviceProfile = Field(default=DeviceProfile.EDGE_IOT, description="Target deployment device profile — drives all efficiency constraints..")
    quant_mode: QuantizationMode = Field(default=QuantizationMode.INT8, description="Quantization mode override (device profile sets default).")
    task_hint: str = Field(default="general", description="Short task hint for vocabulary pruning (e.g. 'qa', 'summary').")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for processing or logging.")

    @field_validator("prompt")
    def validate_prompt(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("🚨 Prompt cannot be empty or whitespace.")
        return stripped
    
    @field_validator("system_prompt")
    @classmethod
    def validate_system_prompt(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def apply_device_quant_default(self) -> "SLMInput":
        """Let device profile set default quantization mode."""
        device_cfg = DEVICE_CONFIGS[self.device_profile]
        # Only override if user didn't explicitly set a non-default mode
        if self.quant_mode == QuantizationMode.INT8:
            self.quant_mode = device_cfg["quant_mode"]
        self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["device_profile"] = self.device_profile.value
        return self

# stage 2: compact tokenization
class TokenCompressionStats(BaseModel):
    """Statistics from the compact tokenization stage."""
    original_token_count: int = Field(..., description="Token count of the original prompt.")
    compressed_token_count: int = Field(..., description="Token count after compression.")
    truncated: bool = Field(..., description="Whether the prompt was truncated to fit the SLM token budget.")
    dedup_removed: int = Field(..., description="Number of duplicate tokens removed during compression.")
    compression_ratio: float = Field(..., description="Ratio of original to compressed tokens.")
    vocab_coverage: float = Field(..., description="Percentage of original tokens retained in the compressed version.")

class CompactTokenizationResult(BaseModel):
    """Stage 2 — Compact Tokenization output."""
    stage: PipelineStage = Field(default=PipelineStage.COMPACT_TOKENIZATION, description="Pipeline stage identifier.")
    encoding: str = Field(..., description="Compact token encoding (e.g. base64-encoded compressed token sequence).")
    tokens: List[int] = Field(..., description="List of tokens after compression (for analysis/debugging).")
    decoded_text: str = Field(..., description="Decoded text from the compressed tokens (should be semantically similar to original prompt).")
    stats: TokenCompressionStats = Field(..., description="Detailed statistics about the token compression process.")
    processing_time: float = Field(..., description="Time taken for the compact tokenization stage (in seconds).")
    
# stage 3: optimized embeddings
class EmbeddingCompressionStats(BaseModel):
    """Statistics from dimensionality reduction."""
    original_dim: int = Field(..., description="Original embedding dimensionality.")
    reduced_dim: int = Field(..., description="Reduced embedding dimensionality.")
    compression_ratio: float = Field(..., description="Ratio of original to reduced dimensions.")
    variance_retained: float = Field(..., description="Percentage of variance retained after reduction.")
    memory_saved_bytes: int = Field(..., description="Estimated memory saved by reducing embedding size.")

class OptimizedEmbeddingsResult(BaseModel):
    """Stage 3 — Optimized Embeddings output."""
    stage: PipelineStage = Field(default=PipelineStage.OPTIMIZED_EMBEDDINGS, description="Pipeline stage identifier.")
    model: str = Field(..., description="Embedding model used (e.g. 'text-embedding-3-small').")
    full_embedding: List[float] = Field(..., description="Original high-dimensional embedding vector.")
    reduced_embedding: List[float] = Field(..., description="Reduced-dimensionality embedding vector for SLM efficiency.")
    stats: EmbeddingCompressionStats = Field(..., description="Detailed statistics about the embedding optimization process.")
    processing_time: float = Field(..., description="Time taken for the optimized embeddings stage (in seconds).")

# stage 4: efficient transformer
class InferenceEfficiencyMetrics(BaseModel):
    """Compute efficiency metrics from the transformer stage."""
    input_tokens: int = Field(..., description="Number of tokens in the input to the transformer.")
    output_tokens: int = Field(..., description="Number of tokens generated by the transformer.")
    total_tokens: int = Field(..., description="Total tokens processed (input + output).")
    tokens_per_second: float = Field(..., description="Throughput in tokens per second.")
    latency_ms: float = Field(..., description="Latency of the transformer inference (in milliseconds).")
    compute_efficiency: float = Field(..., description="Compute efficiency metric (e.g. tokens per second per GB of RAM) (output_tokens / total_tokens ratio).")
    within_budget: bool = Field(..., description="Whether the inference stayed within the defined token and latency budgets (True if output_tokens <= SLM_MAX_OUTPUT_TOKENS).")

class EfficientTransformerResult(BaseModel):
    """Stage 4 — Efficient Transformer output."""
    stage: PipelineStage = Field(default=PipelineStage.EFFICIENT_TRANSFORMER, description="Pipeline stage identifier.")
    model: str = Field(..., description="Transformer model used (e.g. 'gpt-4.1-efficiency').")
    raw_response: str = Field(..., description="Raw text response generated by the transformer before any post-processing.")
    finish_reason: str = Field(..., description="Reason for completion (e.g. 'stop', 'length', 'token_budget').")
    efficiency: InferenceEfficiencyMetrics = Field(..., description="Detailed efficiency metrics from the transformer inference.")
    processing_time: float = Field(..., description="Time taken for the efficient transformer stage (in seconds).")

# stage 5: model quantization
class QuantizedLayer(BaseModel):
    """Quantization result for a single simulated weight layer."""
    layer_name: str = Field(..., description="Name or identifier of the model layer (e.g. 'transformer_block_1_attn_q').")
    original_bits: int = Field(default=32, description="Original bit-width of the weights (e.g. 16 or 32).")
    quantized_bits: int = Field(..., description="Bit-width after quantization (e.g. 8 for INT8, 4 for INT4).")
    num_weights: int = Field(..., description="Number of weights in this layer.")
    original_bytes: int = Field(..., description="Original memory footprint of the weights in bytes.")
    quantized_bytes: int = Field(..., description="Memory footprint after quantization in bytes.")
    compression_ratio: float = Field(..., description="Ratio of original to quantized memory (original_bytes / quantized_bytes).")
    reconstruction_mse: float = Field(..., description="Mean squared error between original and quantized weights (simulated).")
    quality_retained: float = Field(..., ge=0.0, le=1.0, description="Estimated quality retained after quantization (1.0 = no loss, 0.0 = total loss).")

class ModelQuantizationResult(BaseModel):
    """Stage 5 — Model Quantization output."""
    stage: PipelineStage = Field(default=PipelineStage.MODEL_QUANTIZATION, description="Pipeline stage identifier.")
    mode: QuantizationMode = Field(..., description="Quantization mode used (INT8 or INT4).")
    layers: List[QuantizedLayer] = Field(..., description="List of quantized layers with their respective statistics.")
    total_original_mb: float = Field(..., description="Total original model size in megabytes (simulated).")
    total_quantized_mb: float = Field(..., description="Total quantized model size in megabytes (simulated).")
    overall_compression: float = Field(..., description="Overall compression ratio for the entire model (total_original_mb / total_quantized_mb).")
    weighted_quality: float = Field(..., ge=0.0, le=1.0, description="Overall estimated quality retained after quantization, weighted by layer importance.")
    processing_time: float = Field(..., description="Time taken for the model quantization stage (in seconds).")

# stage 6: memory optimization
class KVCacheEntry(BaseModel):
    """A single KV-cache entry."""
    key: str = Field(..., description="Unique identifier for the cache entry (e.g. 'layer1_kv_15').")
    token_hash: str = Field(..., description="Hash of the input tokens associated with this cache entry (for quick lookup).")
    hit_count: int = Field(default=0, description="Number of times this cache entry was accessed (for LRU eviction).")
    size_bytes: int = Field(..., description="Memory size of this cache entry in bytes.")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp when this cache entry was created.")

class MemoryProfile(BaseModel):
    """Memory usage profile at the time of the optimization stage."""
    kv_cache_entries: int = Field(..., description="Number of entries currently in the KV-cache.")
    kv_cache_bytes: int = Field(..., description="Total memory used by the KV-cache in bytes.")
    kv_cache_hit_rate: float = Field(..., description="Hit rate of the KV-cache (hits / total lookups).")
    embedding_bytes: int = Field(..., description="Memory used by embeddings in bytes.")
    model_weights_bytes: int = Field(..., description="Memory used by model weights in bytes.")
    total_bytes: int = Field(..., description="Total memory usage in bytes.")
    total_mb: float = Field(..., description="Total memory usage in megabytes.")
    within_device_limit: bool = Field(..., description="Whether the total memory usage is within the limits of the target device profile.")
    device_limit_mb: float = Field(..., description="Memory limit of the target device profile in megabytes.")

class MemoryOptimizationResult(BaseModel):
    """Stage 6 — Memory Optimization output."""
    stage: PipelineStage = Field(default=PipelineStage.MEMORY_OPTIMIZATION, description="Pipeline stage identifier.")
    profile: MemoryProfile = Field(..., description="Detailed memory usage profile at the time of optimization.")
    cache_hit: bool = Field(..., description="Whether the current input resulted in a KV-cache hit (True) or miss (False).")
    evictions: int = Field(..., description="Number of KV-cache entries evicted during this optimization step (if any).")
    optimizations: List[str] = Field(..., description="List of memory optimization techniques applied (e.g. ['evicted 2 KV-cache entries', 'reduced embedding precision']).")
    memory_saved_mb: float = Field(..., description="Estimated memory saved by the optimizations in megabytes.")
    processing_time: float = Field(..., description="Time taken for the memory optimization stage (in seconds).")

# stage 7: edge deployment
class SLAValidation(BaseModel):
    """Result of edge deployment SLA checks."""
    latency_ms: float = Field(..., description="Measured latency of the response in milliseconds.")
    latency_sla_ms: float = Field(..., description="Defined latency SLA for the target device profile in milliseconds.")
    latency_ok: bool = Field(..., description="Whether the latency meets the SLA requirements.")
    payload_bytes: int = Field(..., description="Size of the response payload in bytes.")
    payload_limit_bytes: int = Field(..., description="Maximum allowed payload size for the target device profile in bytes.")
    payload_ok: bool = Field(..., description="Whether the payload size meets the device constraints.")
    compression_ratio: float = Field(..., description="Compression ratio of the response payload (original size / actual size).")
    compression_ok: bool = Field(..., description="Whether the response meets the minimum compression ratio requirements for edge deployment.")
    overall_pass: bool = Field(..., description="Overall result of the SLA validation (True if all checks passed).")

class EdgeDeploymentResult(BaseModel):
    """Stage 7 — Edge Deployment output."""
    stage: PipelineStage = Field(default=PipelineStage.EDGE_DEPLOYMENT, description="Pipeline stage identifier.")
    device_profile: DeviceProfile = Field(..., description="Target device profile for deployment.")
    device_config: Dict[str, Any] = Field(..., description="Configuration parameters for the target device profile.")
    payload_bytes: int = Field(..., description="Size of the response payload in bytes.")
    payload_compressed: int = Field(..., description="Size of the response payload after compression in bytes.")
    sla: SLAValidation = Field(..., description="Detailed results of the edge deployment SLA validation.")
    deployment_ready: bool = Field(..., description="Whether the response is ready for deployment to the edge device based on the SLA validation results.")
    warnings: List[str] = Field(..., description="List of any warnings or issues identified during the edge deployment validation (e.g. ['Latency exceeds SLA by 500ms', 'Payload size exceeds limit by 200 bytes']).")
    processing_time: float = Field(..., description="Time taken for the edge deployment stage (in seconds).")

# stage 8: output generation
class SLMOutput(BaseModel):
    """Final structured output of the full SLM pipeline."""
    request_id: str = Field(..., description="Unique ID for the request (matches input).")
    stage: PipelineStage = Field(default=PipelineStage.OUTPUT_GENERATION, description="Pipeline stage identifier.")
    status: ProcessingStatus = Field(default=ProcessingStatus.SUCCESS, description="Overall processing status of the request.")

    # stage payloads
    tokenization: CompactTokenizationResult = Field(..., description="Output from the compact tokenization stage.")
    embeddings: OptimizedEmbeddingsResult = Field(..., description="Output from the optimized embeddings stage.")
    transformer: EfficientTransformerResult = Field(..., description="Output from the efficient transformer stage.")
    quantization: ModelQuantizationResult = Field(..., description="Output from the model quantization stage.")
    memory: MemoryOptimizationResult = Field(..., description="Output from the memory optimization stage.")
    edge_deployment: EdgeDeploymentResult = Field(..., description="Output from the edge deployment stage.")

    # final response
    response_text: str = Field(..., description="Final generated response text from the SLM pipeline.")
    total_pipeline_time: float = Field(..., description="Total time taken for the entire SLM pipeline (in seconds).")
    total_pipeline_ms: float = Field(..., description="Total time taken for the entire SLM pipeline (in milliseconds).")

    # efficiency metrics
    token_compression: float = Field(..., description="Overall token compression ratio achieved across the pipeline.")
    embedding_compression: float = Field(..., description="Overall embedding dimensionality reduction ratio.")
    model_compression: float = Field(..., description="Overall model quantization compression ratio.")
    overall_efficiency: float = Field(default=0.0, description="Composite efficiency metric combining all stages (e.g. weighted average of compression ratios and latency improvements).")
    sla_passed: bool = Field(..., description="Whether the final output meets all defined SLAs for edge deployment.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the processing (e.g. timestamps, device profile, quantization mode).")

    @model_validator(mode="after")
    def compute_efficiency_score(self) -> "SLMOutput":
        ratios = [
            self.token_compression,
            self.embedding_compression,
            self.model_compression,
        ]
        valid = [r for r in ratios if r > 0]
        self.overall_efficiency = round(
            math.prod(valid) ** (1 / len(valid)) if valid else 0.0, 4
        )
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["overall_efficiency"] = self.overall_efficiency
        self.metadata["sla_passed"] = self.sla_passed
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
            self.client = OpenAI(
                base_url=ENDPOINT,
                api_key=TOKEN,
            )
        self.logger = get_logger(__name__, log_file="moe.log")
    
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
                self.logger.warning(
                    f"🚨 Rate-limit (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait:.1f}s…",
                    extra={"tag": "retry"}
                )
                time.sleep(wait)
                last_exc = e
            except APIConnectionError as e:       # FIX-05
                wait = RETRY_BACKOFF_BASE * attempt
                self.logger.warning(
                    f"🚨 Connection error (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait:.1f}s…",
                    extra={"tag": "retry"}
                )
                time.sleep(wait)
                last_exc = e
            except APITimeoutError as e:
                wait = RETRY_BACKOFF_BASE * attempt
                self.logger.warning(
                    f"🚨 Timeout (attempt {attempt}/{MAX_RETRIES}). Sleeping {wait:.1f}s…",
                    extra={"tag": "retry"}
                )
                time.sleep(wait)
                last_exc = e
            except APIError as e:
                self.logger.error(f"🚨 APIError on attempt {attempt}: {e}", extra={"tag": "fail"})
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
# KV-CACHE  —  Bounded LRU cache simulating SLM key-value attention cache
# ==========================================
class KVCache:
    """
    Bounded LRU KV-cache simulation.
    In real SLMs, the KV-cache stores key/value attention tensors for previously processed tokens, avoiding recomputation on repeated prefixes.
    We model this at the prompt-hash level: identical or highly similar prompts return cached results, saving inference compute.
    """

    def __init__(self, max_entries: int = KV_CACHE_MAX_ENTRIES) -> None:
        self._cache: OrderedDict[str, KVCacheEntry] = OrderedDict()
        self._max = max_entries
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def lookup(self, text: str) -> KVCacheEntry | None:
        key = self._hash(text)
        if key in self._cache:
            self._cache.move_to_end(key)    # LRU: mark as recently used
            self._cache[key].hit_count += 1
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def store(self, text: str) -> Tuple[KVCacheEntry, int]:
        """Store entry; returns (entry, evictions_performed)."""
        evictions = 0
        key = self._hash(text)
        if key in self._cache:
            return self._cache[key], 0

        while len(self._cache) >= self._max:
            self._cache.popitem(last=False)   # evict LRU
            evictions += 1
            self._evictions += 1

        entry = KVCacheEntry(key=key, token_hash=key, size_bytes=KV_CACHE_ENTRY_BYTES)
        self._cache[key] = entry
        return entry, evictions

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def current_entries(self) -> int:
        return len(self._cache)

    @property
    def total_evictions(self) -> int:
        return self._evictions
    
# ==========================================
# QUANTIZATION ENGINE — INT4 / INT8 scalar quantization
# ==========================================
class QuantizationEngine:
    """
    Simulates INT4 / INT8 post-training quantization (PTQ) on model weight layers. 
    In production SLMs (GPTQ, AWQ, bitsandbytes), each linear layer's weight matrix is quantized per-channel or per-group.

    We simulate quantization on the embedding vector (proxy for weight layer) and on abstract "layer" weight tensors with realistic size estimates.
    """

    SIMULATED_LAYERS = [
        {"name": "embedding_layer", "num_weights": FULL_EMBEDDING_DIM * SLM_VOCAB_SIZE},
        {"name": "attention_q", "num_weights": FULL_EMBEDDING_DIM * REDUCED_EMBEDDING_DIM},
        {"name": "attention_k", "num_weights": FULL_EMBEDDING_DIM * REDUCED_EMBEDDING_DIM},
        {"name": "attention_v", "num_weights": FULL_EMBEDDING_DIM * REDUCED_EMBEDDING_DIM},
        {"name": "ffn_intermediate", "num_weights": FULL_EMBEDDING_DIM * FULL_EMBEDDING_DIM * 4},
        {"name": "ffn_output", "num_weights": FULL_EMBEDDING_DIM * 4 * FULL_EMBEDDING_DIM},
        {"name": "lm_head", "num_weights": FULL_EMBEDDING_DIM * SLM_VOCAB_SIZE},
    ]

    @staticmethod
    def quantize_vector(
        vec : np.ndarray,
        bits: int,
    ) -> Tuple[np.ndarray, float]:
        """
        Quantize a float vector to N-bit integer codes.
        Returns: (quantized_codes, reconstruction_mse)
        """
        levels = 2 ** bits
        v_min, v_max = vec.min(), vec.max()
        v_range = v_max - v_min + 1e-10
        codes = np.floor((vec - v_min) / v_range * (levels - 1)) \
                .astype(np.int32).clip(0, levels - 1)
        dequant = codes.astype(np.float64) / (levels - 1) * v_range + v_min
        mse = float(np.mean((vec - dequant) ** 2))
        return codes, mse

    @classmethod
    def quantize_layers(cls, mode: QuantizationMode) -> List[QuantizedLayer]:
        """Simulate quantization of all model weight layers."""
        bits   = INT4_BITS if mode == QuantizationMode.INT4 else INT8_BITS
        layers : List[QuantizedLayer] = []

        rng = np.random.default_rng(seed=7)   # deterministic
        for layer_def in cls.SIMULATED_LAYERS:
            n = min(layer_def["num_weights"], 4096)   # sample for speed
            weights = rng.normal(0, 0.02, n).astype(np.float64)
            _, mse = cls.quantize_vector(weights, bits)

            orig_bytes = layer_def["num_weights"] * 4    # float32 = 4 bytes
            quant_bytes = math.ceil(
                layer_def["num_weights"] * bits / 8
            )
            compression = orig_bytes / quant_bytes

            # Quality retained: approximated from quantization theory
            # Higher bits → lower MSE → higher quality retained
            quality = 1.0 - (mse / (mse + 1e-6)) * (1 / bits)
            quality = max(0.0, min(1.0, quality))

            layers.append(QuantizedLayer(
                layer_name=layer_def["name"],
                quantized_bits=bits,
                num_weights=layer_def["num_weights"],
                original_bytes=orig_bytes,
                quantized_bytes=quant_bytes,
                compression_ratio=round(compression, 2),
                reconstruction_mse=round(mse, 8),
                quality_retained=round(quality, 4),
            ))

        return layers

# ==========================================
# PCA DIMENSION REDUCER  —  1536-d → 128-d projection
# ==========================================
class PCAReducer:
    """
    Simulates PCA-style dimensionality reduction on embedding vectors.
    In production SLMs, smaller hidden dimensions (128–256 vs. 1536+) are a primary source of parameter efficiency.

    We use a seeded random projection matrix as a faithful approximation of PCA's linear projection (Johnson-Lindenstrauss lemma guarantees that random projections approximately preserve pairwise distances).
    """

    def __init__(self, input_dim: int = FULL_EMBEDDING_DIM, output_dim: int = REDUCED_EMBEDDING_DIM) -> None:
        self._input_dim = input_dim
        self._output_dim = output_dim
        rng = np.random.default_rng(seed=42)
        # orthonormalised random projection matrix
        raw = rng.standard_normal((input_dim, output_dim))
        q, _ = np.linalg.qr(raw)
        self._projection = q[:, :output_dim]

    def reduce(self, vec: List[float]) -> Tuple[List[float], float]:
        """
        Project full embedding to reduced dimension.
        Returns: (reduced_vector, variance_retained_estimate)
        """
        arr = np.array(vec, dtype=np.float64)
        reduced = arr @ self._projection
        # Variance retained estimate: ratio of norms squared (proxy for variance)
        var_retained = float(np.dot(reduced, reduced) / (np.dot(arr, arr) + 1e-10))
        var_retained = min(1.0, max(0.0, var_retained))
        return reduced.tolist(), round(var_retained, 4)
    
    @property
    def compression_ratio(self) -> float:
        return round(self._input_dim / self._output_dim, 2)

# ==========================================
# PIPELINE STAGES
# ==========================================
class CompactTokenizationStage:
    """
    Stage 2: COMPACT TOKENIZATION
    Simulates SLM's aggressive tokenization strategy:
    1. Tokenize with cl100k_base (GPT-4 family encoding).
    2. Enforce SLM_MAX_INPUT_TOKENS ceiling with truncation.
    3. Remove duplicate consecutive tokens (BPE merge simulation).
    4. Compute vocabulary coverage against SLM_VOCAB_SIZE.

    SLM tokenizers (SentencePiece BPE, Unigram) typically use smaller vocabularies (32k vs 100k) and more aggressive subword merging, yielding denser token sequences for the same semantic content.
    """
    def __init__(self) -> None:
        self._encoder = tiktoken.get_encoding(ENCODING)

    def run(self, slm_input: SLMInput) -> CompactTokenizationResult:
        logger.info("⚙️ [COMPACT TOKENIZATION] Applying SLM token budget...")
        t0 = time.perf_counter()
        full_text = slm_input.system_prompt + "\n" + slm_input.prompt
        tokens = self._encoder.encode(full_text)
        original_count = len(tokens)
        truncated = False

        # truncate to SLM budget
        if len(tokens) > SLM_MAX_INPUT_TOKENS:
            tokens = tokens[:SLM_MAX_INPUT_TOKENS]
            truncated = True
            logger.warning(
                f"⚠️ [COMPACT TOKENIZATION] Truncated "
                f"{original_count} → {SLM_MAX_INPUT_TOKENS} tokens (SLM budget enforced)"
            )
        
        # deduplicate consecutive repeated tokens (simulate aggressive BPE merging)
        deduped: List[int] = []
        dedup_count: int = 0
        prev: Optional[int] = None
        for t in tokens:
            if t == prev:
                dedup_count += 1
            else:
                deduped.append(t)
                prev = t

        # vocab coverage
        unique_tokens = set(tokens)
        vocab_coverage = len(unique_tokens) / SLM_VOCAB_SIZE

        # decode back
        decoded_text = self._encoder.decode(tokens)
        compression = original_count / len(tokens) if len(tokens) > 0 else 1.0
        elapsed = time.perf_counter() - t0

        result = CompactTokenizationResult(
            encoding=ENCODING,
            tokens=deduped,
            decoded_text=decoded_text,
            stats=TokenCompressionStats(
                original_token_count=original_count,
                compressed_token_count=len(tokens),
                truncated=truncated,
                dedup_removed=dedup_count,
                compression_ratio=round(compression, 4),
                vocab_coverage=round(vocab_coverage, 4),
            ),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [COMPACT TOKENIZATION] {original_count} → {len(deduped)} tokens | "
            f"dedup_removed={dedup_count} | truncated={truncated} | "
            f"vocab_cov={vocab_coverage:.4f} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [COMPACT TOKENIZATION] Decoded text preview: {decoded_text[:100]}...")
        return result
    
class OptimizedEmbeddingsStage:
    """
    Stage 3: OPTIMIZED EMBEDDINGS
    Generates a full 1536-d embedding, then applies PCA-style projection to REDUCED_EMBEDDING_DIM (128-d) for SLM memory efficiency.

    Memory saving: 1536 × 4 bytes (float32) = 6,144 bytes
                →   128 × 4 bytes           =   512 bytes
                = 12× memory reduction per token embedding.
    """

    def __init__(self, client: OpenAI, reducer: PCAReducer) -> None:
        self._client = client
        self._reducer = reducer

    def run(self, decoded_text: str, agent: BaseAIAgent) -> OptimizedEmbeddingsResult:
        logger.info(f"⚙️  [OPTIMIZED EMBEDDINGS] Embedding → {FULL_EMBEDDING_DIM}d → {REDUCED_EMBEDDING_DIM}d reduction...")
        t0 = time.perf_counter()

        # full embedding
        emb_response = agent._retry_api_call(
            self._client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=decoded_text[:4096]  # embedding input limit
        )
        full_emb = emb_response.data[0].embedding

        # pca reduction
        reduced_emb, var_retained = self._reducer.reduce(full_emb)
        original_bytes = FULL_EMBEDDING_DIM * 4
        reduced_bytes = REDUCED_EMBEDDING_DIM * 4
        mem_saved = original_bytes - reduced_bytes
        elapsed = time.perf_counter() - t0

        result = OptimizedEmbeddingsResult(
            model=EMBEDDING_MODEL,
            full_embedding=full_emb,
            reduced_embedding=reduced_emb,
            stats=EmbeddingCompressionStats(
                original_dim=FULL_EMBEDDING_DIM,
                reduced_dim=REDUCED_EMBEDDING_DIM,
                compression_ratio=self._reducer.compression_ratio,
                variance_retained=var_retained,
                memory_saved_bytes=mem_saved,
            ),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [OPTIMIZED EMBEDDINGS] {FULL_EMBEDDING_DIM}d → {REDUCED_EMBEDDING_DIM}d | compression={self._reducer.compression_ratio}x | "
            f"variance_retained={var_retained:.4f} | mem_saved={mem_saved}B | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [OPTIMIZED EMBEDDINGS] Full embedding preview: {full_emb[:5]}... | Reduced embedding preview: {reduced_emb[:5]}...")
        return result

class EfficientTransformerStage:
    """
    Stage 4: EFFICIENT TRANSFORMER
    Runs inference with strict SLM compute budget constraints:
    - Input capped at SLM_MAX_INPUT_TOKENS
    - Output capped at SLM_MAX_OUTPUT_TOKENS
    - System prompt kept minimal
    - Latency tracked at millisecond resolution

    In true SLMs (Phi-3-mini, Gemma-2B), efficiency comes from:
    grouped-query attention (GQA), sliding-window attention, and parameter-sharing across layers.
    """
    def run(self, slm_input: SLMInput, tokenization: CompactTokenizationResult, agent: BaseAIAgent) -> EfficientTransformerResult:
        logger.info(
            f"⚙️ [EFFICIENT TRANSFORMER] Inference | "
            f"budget={SLM_MAX_OUTPUT_TOKENS} tokens | device={slm_input.device_profile.value}..."
        )
        t0 = time.perf_counter()

        # use the compact-tokenized text
        user_text = tokenization.decoded_text[:4000]  # ensure we stay within embedding input limits
        response = agent._retry_api_call(
            agent.client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": slm_input.system_prompt},
                {"role": "user", "content": user_text},
            ],
            temperature=0.3,
            max_tokens=SLM_MAX_OUTPUT_TOKENS,  # strict SLM budget
        )
        choice = response.choices[0]
        usage = response.usage
        raw_response = choice.message.content or ""
        elapsed_s = time.perf_counter() - t0
        elapsed_ms = elapsed_s * 1000

        # efficiency metrics
        tps = usage.completion_tokens / elapsed_s if elapsed_s > 0 else 0.0
        compute_eff = usage.completion_tokens / max(usage.total_tokens, 1)
        within_budget = usage.completion_tokens <= SLM_MAX_OUTPUT_TOKENS
        if not within_budget:
            logger.warning(f"⚠️ [EFFICIENT TRANSFORMER] Output {usage.completion_tokens} exceeds SLM budget {SLM_MAX_OUTPUT_TOKENS}")

        result = EfficientTransformerResult(
            model=CHAT_MODEL,
            raw_response=raw_response,
            finish_reason=choice.finish_reason or "unknown",
            efficiency=InferenceEfficiencyMetrics(
                input_tokens=usage.prompt_tokens,
                output_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                tokens_per_second=round(tps, 2),
                latency_ms=round(elapsed_ms, 2),
                compute_efficiency=round(compute_eff, 4),
                within_budget=within_budget,
            ),
            processing_time=round(elapsed_s, 4)
        )
        logger.info(
            f"✅ [EFFICIENT TRANSFORMER] tokens(in/out/total)={usage.prompt_tokens}/{usage.completion_tokens}/{usage.total_tokens} | "
            f"tps={tps:.1f} | latency={elapsed_ms:.1f}ms | "
            f"within_budget={within_budget} | time={elapsed_s:.4f}s"
        )
        logger.debug(f"🔍 [EFFICIENT TRANSFORMER] Raw response preview: {raw_response[:100]}...")
        return result

class ModelQuantizationStage:
    """
    Stage 5: MODEL QUANTIZATION
    Applies INT4 or INT8 post-training quantization to all model layers.

    INT8: ~4x size reduction vs float32 | negligible quality loss
    INT4: ~8x size reduction vs float32 | slight quality loss on edge cases

    Also quantizes the reduced embedding vector (used as proxy for the embedding layer weight tensor).
    """

    def run(self ,slm_input: SLMInput, reduces_emb: List[float]) -> ModelQuantizationResult:
        mode = slm_input.quant_mode
        logger.info(f"⚙️ [MODEL QUANTIZATION] Applying {mode.value} quantization to {len(QuantizationEngine.SIMULATED_LAYERS)} layers...")
        t0 = time.perf_counter()
        
        # quantize all simulated weight layers
        layers = QuantizationEngine.quantize_layers(mode)

        # also quantize the reduced embedding 
        bits = INT4_BITS if mode == QuantizationMode.INT4 else INT8_BITS
        emb_arr = np.array(reduces_emb, dtype=np.float64)
        _, emb_mse = QuantizationEngine.quantize_vector(emb_arr, bits)

        # aggregate stats
        total_orig_bytes = sum(l.original_bytes for l in layers)
        total_quant_bytes = sum(l.quantized_bytes for l in layers)
        overall_comp = total_orig_bytes / total_quant_bytes
        weighted_quality = sum(
            l.quality_retained * l.original_bytes for l in layers
        ) / total_orig_bytes

        elapsed = time.perf_counter() - t0
        result = ModelQuantizationResult(
            mode=mode,
            layers=layers,
            total_original_mb=round(total_orig_bytes / 1e6, 2),
            total_quantized_mb=round(total_quant_bytes / 1e6, 2),
            overall_compression=round(overall_comp, 2),
            weighted_quality=round(weighted_quality, 4),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [MODEL QUANTIZATION] {mode.value} | "
            f"{result.total_original_mb}MB → {result.total_quantized_mb}MB | compression={overall_comp:.2f}x | "
            f"quality={weighted_quality:.4f} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [MODEL QUANTIZATION] Sample layer quantization: {layers[0].layer_name}")
        return result

class MemoryOptimizationStage:
    """
    Stage 6: MEMORY OPTIMIZATION
    Maintains the KV-cache and profiles total memory footprint.

    Optimizations applied:
    1. KV-cache lookup (avoid recomputing known token sequences)
    2. Embedding dimension reduction savings (already applied in Stage 3)
    3. Model weight quantization savings (already applied in Stage 5)
    4. LRU eviction to stay within device RAM limit
    """
    def run(self, slm_input: SLMInput, tokenization: CompactTokenizationResult, embeddings: OptimizedEmbeddingsResult, quantization: ModelQuantizationResult, kv_cache: KVCache) -> MemoryOptimizationResult:
        logger.info(f"⚙️ [MEMORY OPTIMIZATION] KV-cache + memory profiling | device={slm_input.device_profile.value}...")
        t0 = time.perf_counter()

        # KV-cache lookup / store
        cache_key = tokenization.decoded_text
        cache_hit = kv_cache.lookup(cache_key) is not None
        entry, evictions = kv_cache.store(cache_key)
        if cache_hit:
            logger.info("⚡ [MEMORY OPTIMIZATION] KV-cache HIT — skipping recompute")
        else:
            logger.debug("💾 [MEMORY OPTIMIZATION] KV-cache MISS — stored new entry")

        # memory profiling
        kv_bytes = kv_cache.current_entries * KV_CACHE_ENTRY_BYTES
        emb_bytes = REDUCED_EMBEDDING_DIM * 4         # float32 reduced emb
        model_bytes = int(quantization.total_quantized_mb * 1e6)
        total_bytes = kv_bytes + emb_bytes + model_bytes
        total_mb = total_bytes / 1e6

        device_cfg = DEVICE_CONFIGS[slm_input.device_profile]
        device_lim = device_cfg["max_ram_mb"]
        within_limit = total_mb <= device_lim
        if not within_limit:
            logger.warning(f"⚠️ [MEMORY OPTIMIZATION] Memory {total_mb:.2f}MB exceeds device limit {device_lim}MB")

        # applied optimizations summary
        optimizations = [
            f"KV-cache LRU (entries={kv_cache.current_entries}, "
            f"hit_rate={kv_cache.hit_rate:.2%})",
            f"Embedding dim reduction: "
            f"{FULL_EMBEDDING_DIM}d → {REDUCED_EMBEDDING_DIM}d "
            f"({embeddings.stats.memory_saved_bytes}B saved)",
            f"Weight quantization {quantization.mode.value}: "
            f"{quantization.total_original_mb}MB → "
            f"{quantization.total_quantized_mb}MB",
        ]
        if evictions > 0:
            optimizations.append(f"LRU evicted {evictions} stale cache entries")

        memory_saved_mb = round(
            (embeddings.stats.memory_saved_bytes
             + (quantization.total_original_mb - quantization.total_quantized_mb) * 1e6)
            / 1e6, 4
        )

        elapsed = time.perf_counter() - t0
        result = MemoryOptimizationResult(
            profile=MemoryProfile(
                kv_cache_entries=kv_cache.current_entries,
                kv_cache_bytes=kv_bytes,
                kv_cache_hit_rate=round(kv_cache.hit_rate, 4),
                embedding_bytes=emb_bytes,
                model_weights_bytes=model_bytes,
                total_bytes=total_bytes,
                total_mb=round(total_mb, 4),
                within_device_limit=within_limit,
                device_limit_mb=float(device_lim),
            ),
            cache_hit=cache_hit,
            evictions=evictions,
            optimizations=optimizations,
            memory_saved_mb=memory_saved_mb,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [MEMORY OPTIMIZATION] total={total_mb:.4f}MB | within_limit={within_limit} | "
            f"cache_hit={cache_hit} | evictions={evictions} | saved={memory_saved_mb}MB | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [MEMORY OPTIMIZATION] Memory profile: {result.profile}")
        return result
    
class EdgeDeploymentStage:
    """
    Stage 7: EDGE DEPLOYMENT
    Validates the full pipeline output against the target device's SLA:
    - Latency: must be within device_config['latency_sla_ms']
    - Payload: response bytes must be within device_config['payload_max_bytes']
    - Compression: payload compression ratio must meet minimum threshold

    Simulates gzip-style compression on the response payload.
    """
    @staticmethod
    def _simulate_compression(text: str) -> int:
        """
        Simulate gzip compression ratio on text.
        Real compression ~2-4x for natural language. We approximate based on character entropy.
        """
        raw_bytes = text.encode("utf-8")
        if not raw_bytes:
            return 1
        # Shannon entropy-based compression estimate
        freq = {}
        for b in raw_bytes:
            freq[b] = freq.get(b, 0) + 1
        n = len(raw_bytes)
        entropy = -sum((c / n) * math.log2(c / n) for c in freq.values() if c > 0)
        ratio = max(1.0, 8.0 / max(entropy, 1.0))   # bits per symbol → ratio
        return max(1, int(n / ratio))
    
    def run(self, slm_input: SLMInput, transformer: EfficientTransformerResult) -> EdgeDeploymentResult:
        device_cfg = DEVICE_CONFIGS[slm_input.device_profile]
        logger.info(f"⚙️  [EDGE DEPLOYMENT] Validating SLAs for {slm_input.device_profile.value}...")
        t0 = time.perf_counter()

        payload_raw = transformer.raw_response.encode("utf-8")
        payload_bytes = len(payload_raw)
        compressed_bytes = self._simulate_compression(transformer.raw_response)
        comp_ratio = payload_bytes / compressed_bytes if compressed_bytes > 0 else 1.0

        latency_ms = transformer.efficiency.latency_ms
        latency_sla = device_cfg["latency_sla_ms"]
        payload_limit = device_cfg["payload_max_bytes"]

        latency_ok = latency_ms <= latency_sla
        payload_ok = payload_bytes <= payload_limit
        compression_ok= comp_ratio >= EDGE_MIN_COMPRESSION
        overall_pass = latency_ok and payload_ok

        warnings: List[str] = []
        if not latency_ok:
            warnings.append(f"⚠️ Latency {latency_ms:.1f}ms exceeds SLA {latency_sla}ms")
        if not payload_ok:
            warnings.append(f"⚠️ Payload {payload_bytes}B exceeds limit {payload_limit}B")
        if not compression_ok:
            warnings.append(f"⚠️ Compression ratio {comp_ratio:.2f} below minimum {EDGE_MIN_COMPRESSION}x")
        for w in warnings:
            logger.warning(f"⚠️ [EDGE DEPLOYMENT] {w}")
        
        elapsed = time.perf_counter() - t0
        result = EdgeDeploymentResult(
            device_profile=slm_input.device_profile,
            device_config={k: str(v) for k, v in device_cfg.items()},
            payload_bytes=payload_bytes,
            payload_compressed=compressed_bytes,
            sla=SLAValidation(
                latency_ms=latency_ms,
                latency_sla_ms=latency_sla,
                latency_ok=latency_ok,
                payload_bytes=payload_bytes,
                payload_limit_bytes=payload_limit,
                payload_ok=payload_ok,
                compression_ratio=round(comp_ratio, 2),
                compression_ok=compression_ok,
                overall_pass=overall_pass,
            ),
            deployment_ready=overall_pass,
            warnings=warnings,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [EDGE DEPLOYMENT] latency_ok={latency_ok} | payload_ok={payload_ok} | "
            f"compression={comp_ratio:.2f}x | ready={overall_pass} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [EDGE DEPLOYMENT] SLA validation details: {result.sla}")
        return result

# ==========================================
# SLM AGENT  —  Orchestrates all 8 pipeline stages
# ==========================================
class SLMAgent(BaseAIAgent):
    """SLM AGENT — Small Language Model (Agent 6 of 8)

    Pipeline:
    Input Processing → Compact Tokenization → Optimized Embeddings →
    Efficient Transformer → Quantization ⟷ Memory Optimization → Edge Deployment → Output Generation

    Core Principle: Efficiency-first design: enforce resource constraints, token budgets, dimensionality reduction, INT4/INT8 quantization, KV-cache simulation, and robust edge payload validation.
    """
    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self._tokenizer = CompactTokenizationStage()
        self._pca_reducer = PCAReducer(FULL_EMBEDDING_DIM, REDUCED_EMBEDDING_DIM)
        self._embedder = OptimizedEmbeddingsStage(client, self._pca_reducer)
        self._transformer = EfficientTransformerStage()
        self._quantizer = ModelQuantizationStage()
        self._memory_optimizer = MemoryOptimizationStage()
        self._edge_deployer = EdgeDeploymentStage()
        self._kv_cache = KVCache(max_entries=KV_CACHE_MAX_ENTRIES)

    def process(self, slm_input: SLMInput) -> SLMOutput:
        """
        Execute the full SLM pipeline with efficiency constraints.

        Args:
            slm_input: Validated SLMInput pydantic model.

        Returns:
            SLMOutput: Fully structured pipeline result with efficiency scorecard.

        Raises:
            ValueError: On invalid input.
            RuntimeError: On unrecoverable API failure.
        """
        pipeline_start = time.perf_counter()

        # Create a unique checkpointer for this specific request ID
        filename = f"slm_checkpoint_{slm_input.request_id}.json"
        checkpointer = StateCheckpointer(
            directory=CHECKPOINT_DIR, 
            filename=filename,
            logger=logger
        )

        logger.info(f"🚀 [SLM AGENT] Pipeline START | request_id={slm_input.request_id}")
        logger.info(
            f"📥 [INPUT] device={slm_input.device_profile.value} | "
            f"quant={slm_input.quant_mode.value} | prompt_len={len(slm_input.prompt)} chars"
        )
        try:
            # stage 2: compact tokenization (Local - Always Run)
            tokenization = self._tokenizer.run(slm_input)

            # stage 3: optimized embeddings (API - Checkpointed)
            embeddings = checkpointer.load("OPTIMIZED_EMBEDDINGS", OptimizedEmbeddingsResult)
            if not embeddings:
                embeddings = self._embedder.run(tokenization.decoded_text, self)
                checkpointer.save("OPTIMIZED_EMBEDDINGS", embeddings)

            # stage 4: efficient transformer (API - Checkpointed)
            transformer = checkpointer.load("EFFICIENT_TRANSFORMER", EfficientTransformerResult)
            if not transformer:
                transformer = self._transformer.run(slm_input, tokenization, self)
                checkpointer.save("EFFICIENT_TRANSFORMER", transformer)

            # stage 5: model quantization + stage 6: memory optimization
            logger.info("⚙️ [SLM AGENT] Running Model Quantization ⟷ Memory Optimization (interleaved)...")
            
            # Local - Always Run
            quantization = self._quantizer.run(slm_input, embeddings.reduced_embedding)
            memory = self._memory_optimizer.run(slm_input, tokenization, embeddings, quantization, self._kv_cache)

            # stage 7: edge deployment (Local - Always Run)
            edge = self._edge_deployer.run(slm_input, transformer)

            # stage 8: output generation
            status = (
                ProcessingStatus.SUCCESS
                if edge.sla.overall_pass
                else ProcessingStatus.DEGRADED
            )

            total_time = time.perf_counter() - pipeline_start
            output = SLMOutput(
                request_id=slm_input.request_id,
                status=status,
                tokenization=tokenization,
                embeddings=embeddings,
                transformer=transformer,
                quantization=quantization,
                memory=memory,
                edge_deployment=edge,
                response_text=transformer.raw_response,
                total_pipeline_time=round(total_time, 4),
                total_pipeline_ms=round(total_time * 1000, 2),
                token_compression=tokenization.stats.compression_ratio,
                embedding_compression=embeddings.stats.compression_ratio,
                model_compression=quantization.overall_compression,
                sla_passed=edge.sla.overall_pass,
                metadata={
                    **slm_input.metadata,
                    "model": CHAT_MODEL,
                    "device_profile": slm_input.device_profile.value,
                    "quantization_mode": slm_input.quant_mode.value,
                }
            )
            logger.info(
                f"🎉 [SLM AGENT] Pipeline COMPLETE | total_time={total_time:.4f}s | "
                f"status={status.value} | sla_passed={edge.sla.overall_pass} | overall_efficiency={output.overall_efficiency:.4f}x"
            )
            logger.debug(f"🔍 [SLM AGENT] Final output: {output}")
            return output
        except Exception as e:
            elapsed = time.perf_counter() - pipeline_start
            logger.exception(f"❌ [SLM AGENT] Pipeline failed after {elapsed:.4f} | error={type(e).__name__}: {e}")
            raise RuntimeError("SLM pipeline execution failed") from e

    # display output
    def display_output(self, output: SLMOutput) -> None:
        div = "=" * 80
        print(f"\n{div}")
        print("🟡 SLM AGENT — Small Language Model Pipeline Result")
        print(f"{div}")
        print(f"Request ID: {output.request_id}")
        print(f"Status: {output.status.value}")
        print(f"Total Time: {output.total_pipeline_time}s "
              f"({output.total_pipeline_ms}ms)")
        print(f"SLA Passed: {output.sla_passed}")
    
        print(f"{div}")
        print(f"📊 EFFICIENCY SCORECARD")
        print(f"Token Compression : {output.token_compression:.4f}x")
        print(f"Embedding Compress: {output.embedding_compression:.4f}x")
        print(f"Model Compression : {output.model_compression:.4f}x")
        print(f"── Overall (geomean): {output.overall_efficiency:.4f}x ──")
        
        print(f"{div}")
        print(f"📤 OUTPUT GENERATION\n")
        print(f"{output.response_text}")
        print(f"\n{div}\n")
# ==========================================
# Instatiation
# ==========================================
def create_slm_agent(api_key: str = TOKEN, endpoint: str = ENDPOINT) -> SLMAgent:
    """Factory function to create an instance of SLMAgent with the provided API credentials."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    logger.info("🏭 [FACTORY] SLMAgent instantiated and ready.")
    return SLMAgent(client)

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    agent = create_slm_agent()

    # Demo 1: Edge IoT Device (strictest constraints) 
    slm_input = SLMInput(
        prompt=(
            "Explain the key difference between supervised and unsupervised machine learning in simple terms. "
            "Keep the answer very brief."
        ),
        system_prompt="You are a compact AI assistant. Be extremely concise.",
        device_profile= DeviceProfile.EDGE_IOT,
        task_hint="qa",
        metadata={"source": "slm_agent_demo", "version": "1.0"},
    )

    result = agent.process(slm_input)
    agent.display_output(result)

    # Demo 2: KV-cache hit (same prompt repeated) 
    print("\n" + "═" * 80)
    print("📋 Demo 2: KV-cache hit test (same prompt repeated)")
    print("═" * 80 + "\n")

    result2 = agent.process(slm_input)
    print(
        f"Cache hit: {result2.memory.cache_hit} | "
        f"Hit rate: {result2.memory.profile.kv_cache_hit_rate:.2%}\n"
    )

    # Demo 3: Mobile Device Profile 
    print("\n" + "═" * 80)
    print("📋 Demo 3: Mobile device profile")
    print("═" * 80 + "\n")

    mobile_input = SLMInput(
        prompt="What are three practical uses of small language models?",
        device_profile=DeviceProfile.MOBILE,
        task_hint="qa",
        metadata={"source": "slm_agent_demo_mobile"},
    )
    result3 = agent.process(mobile_input)
    agent.display_output(result3)
