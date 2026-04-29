import os, uuid, time, re, json, base64, hashlib, mimetypes
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from urllib.request import urlopen
from urllib.error import URLError

from dotenv import load_dotenv
load_dotenv()

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="vlm.log")

# ==========================================
# Variable Configuration
# ==========================================
from utils.state_checkpointer import StateCheckpointer
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
checkpointer = StateCheckpointer(
    directory=CHECKPOINT_DIR, 
    filename="vlm_checkpoint.json",
    logger=logger
)

TOKEN = os.environ['GITHUB_TOKEN']
ENDPOINT = os.environ['GITHUB_ENDPOINT']
CHAT_MODEL = os.environ['GITHUB_MODEL_NAME']
EMBEDDING_MODEL = os.environ['GITHUB_EMBED_NAME']
EMBEDDING_DIMENSIONS = 1536

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
MAX_IMAGE_BYTES = 20 * 1024 * 1024  # 20 MB hard limit
SUPPORTED_MIME_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}

# Vision detail level: "low" (85 tokens) | "high" (full patch grid)
VISION_DETAIL_LEVEL = "high"

# Projection alignment threshold
PROJECTION_ALIGN_THRESHOLD = 0.3  # min cosine similarity for valid alignment

# prompt templates
VISION_ENCODER_SYSTEM_PROMPT = """
You are a vision encoder for a Vision-Language Model.
Perform detailed visual feature extraction from the image.
Respond with structured JSON only — no prose.
"""

VISION_ENCODER_PROMPT_TEMPLATE = """
Extract a complete visual feature map from this image.

Respond with:
{
  "scene_type": "<indoor|outdoor|abstract|document|other>",
  "dominant_colors": ["<color1>", "<color2>", "<color3>"],
  "detected_objects": ["<obj1>", "<obj2>"],
  "spatial_layout": "<description of spatial arrangement>",
  "visual_regions": [
    {
      "region_id": "R001",
      "label": "<region label>",
      "description": "<detailed description>",
      "spatial_hint": "<top-left|top-right|center|bottom-left|bottom-right|full>",
      "salience": <0.0-1.0>
    }
  ],
  "texture_desc": "<texture description>",
  "lighting_desc": "<lighting description>",
  "estimated_depth": "<flat|shallow|deep>",
  "contains_text": <true|false>,
  "contains_faces": <true|false>,
  "visual_complexity": <0.0-1.0>
}
"""

MULTIMODAL_PROCESSOR_PROMPT_TEMPLATE = """
Text Query: {text_query}
Key Entities: {key_entities}

Visual Regions:
{regions_desc}

Shared Concepts: {visual_text_overlap}

Respond with:
{{
  "cross_attention": [
    {{ "text_token": "<entity>", "visual_region": "<R00x>", "weight": <0.0-1.0> }}
  ],
  "grounded_concepts": ["<concept anchored in both>"],
  "multimodal_context": "<fused representation in one paragraph>",
  "visual_text_conflicts": ["<conflict if any, else empty list>"]
}}
"""

LANGUAGE_MODEL_SYSTEM_PROMPT = """
You are a Vision-Language Model specialising in {task_type}.
Your response must be grounded in specific visual evidence from the image.
Reference specific objects, regions, colours, or spatial relationships you can actually see.
Do not hallucinate details not present in the image.
"""

# ==========================================
# ENUMS
# ==========================================
class PipelineStage(str, Enum):
    IMAGE_INPUT = "IMAGE_INPUT"
    TEXT_INPUT = "TEXT_INPUT"
    VISION_ENCODER = "VISION_ENCODER"
    TEXT_ENCODER = "TEXT_ENCODER"
    PROJECTION_INTERFACE = "PROJECTION_INTERFACE"
    MULTIMODAL_PROCESSOR = "MULTIMODAL_PROCESSOR"
    LANGUAGE_MODEL = "LANGUAGE_MODEL"
    OUTPUT_GENERATION = "OUTPUT_GENERATION"

class ImageSource(str, Enum):
    FILE = "FILE"
    URL = "URL"
    BASE64 = "BASE64"
    BYTES = "BYTES"

class TaskType(str, Enum):
    VISUAL_QA = "VISUAL_QA"
    IMAGE_CAPTIONING = "IMAGE_CAPTIONING"
    OBJECT_DETECTION = "OBJECT_DETECTION"
    SCENE_UNDERSTANDING = "SCENE_UNDERSTANDING"
    TEXT_EXTRACTION = "TEXT_EXTRACTION"
    VISUAL_REASONING = "VISUAL_REASONING"
    COMPARISON = "COMPARISON"
    GROUNDING = "GROUNDING"

class ProcessingStatus(str, Enum):
    PENDING  = "PENDING"
    SUCCESS  = "SUCCESS"
    FAILED   = "FAILED"

# ==========================================
# PYDANTIC MODELS
# ==========================================
# stage: image input + text input
class ImagePayload(BaseModel):
    """
    Validated image payload for the VLM pipeline.
    Accepts file path, URL, raw bytes, or pre-encoded base64.
    """
    source_type: ImageSource
    source: str = Field(..., description="File path, URL, or base64 string depending on source_type.")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the image (e.g., image/jpeg).") 
    size_bytes: int = Field(default=0, ge=0, le=20, description="Size of the image in bytes, must be <= 20MB.")
    sha256: str = Field(default="", description="SHA-256 hash of the image content for integrity verification.")
    width_hint: Optional[int] = Field(default=None, ge=1, description="Optional width hint for the image in pixels.")
    height_hint: Optional[int] = Field(default=None, ge=1, description="Optional height hint for the image in pixels.")

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v: str) -> str:
        if v not in SUPPORTED_MIME_TYPES:
            raise ValueError(f"Unsupported MIME type: {v}. Supported types: {SUPPORTED_MIME_TYPES}")
        return v

class VLMInput(BaseModel):
    """Combined image + text input to the VLM pipeline."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for the request.")
    image: ImagePayload
    text_query: str = Field(..., min_length=1, max_length=8000, description="User's textual query or instruction about the image.")
    task_type: TaskType = Field(default=TaskType.VISUAL_QA, description="Type of task to perform on the image.")
    detail_level: str = Field(default=VISION_DETAIL_LEVEL, pattern="^(low|high)$", description="Level of visual detail to extract (e.g., 'low' or 'high').")
    max_tokens: int = Field(default=1024, ge=64, le=4096, description="Max tokens for the language model response.")
    temperature: float = Field(default=0.3, ge=0.0, le=2.0, description="Temperature for language model generation.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for processing or logging.")

    @field_validator("text_query")
    @classmethod
    def validate_text_query(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("🚨 Text query cannot be empty.")
        return v.strip()
    
    @model_validator(mode="after")
    def stamp_metadata(self) -> "VLMInput":
        self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["task_type"] = self.task_type.value
        return self
    
# stage: vision encoder
class VisualRegion(BaseModel):
    """A semantically meaningful region detected by the vision encoder."""
    region_id: str = Field(description="Unique identifier for the visual region.")
    label: str = Field(description="Semantic label for the region (e.g., 'cat', 'tree').")
    description: str = Field(description="Detailed description of the region's visual features.")
    spatial_hint: str = Field(description="Relative spatial hint (e.g., 'top-left', 'center', 'bottom-right').")
    salience: float = Field(ge=0.0, le=1.0, description="Salience score indicating the importance of the region for the given query.")

class VisualFeatureMap(BaseModel):
    """Structured visual features extracted by the vision encoder."""
    scene_type: str = Field(description="High-level scene classification (e.g., 'indoor', 'outdoor').")
    dominant_colors: List[str] = Field(description="List of dominant colors in the image (e.g., ['red', 'green', 'blue']).")
    detected_objects: List[str] = Field(description="List of detected objects/regions in the image.") 
    spatial_layout: str = Field(description="Description of the spatial layout of the scene (e.g., 'a cat sitting on a sofa in a living room').")
    visual_regions: List[VisualRegion] = Field(description="List of semantically meaningful regions detected in the image.")
    texture_desc: str = Field(description="Description of the overall texture and visual complexity of the image (e.g., 'smooth, low-texture' or 'highly detailed, textured').")
    lighting_desc: str = Field(description="Description of the lighting conditions in the image (e.g., 'well-lit with natural light' or 'dimly lit with harsh shadows').")
    estimated_depth: str = Field(description="Estimated depth information or 3D structure cues in the scene (e.g., 'foreground objects are sharp and detailed, background is blurry').")
    contains_text: bool = Field(description="Whether the image contains visible text that may be relevant to the query.")
    contains_faces: bool = Field(description="Whether the image contains human faces that may be relevant to the query.")
    visual_complexity: float = Field(ge=0.0, le=1.0, description="A score indicating the overall visual complexity of the image, which may impact processing strategies.")

class VisionEncoderResult(BaseModel):
    """Stage: Vision Encoder output."""
    stage: PipelineStage = Field(default=PipelineStage.VISION_ENCODER, description="Pipeline stage identifier.")
    model: str = Field(description="Name of the vision encoder model used.")
    detail_level: str = Field(description="Level of visual detail extracted (e.g., 'low' or 'high').")
    feature_map: VisualFeatureMap = Field(description="Structured visual features extracted from the image.")
    patch_token_est: int = Field(ge=0, description="Estimated number of tokens required to represent the visual features in text form.")
    processing_time: float = Field(description="Time taken to process the image and extract visual features, in seconds.")

# stage: text encoder
class TextEncoderResult(BaseModel):
    """Stage: Text Encoder output."""
    stage: PipelineStage = Field(default=PipelineStage.TEXT_ENCODER, description="Pipeline stage identifier.")
    model: str = Field(description="Name of the text encoder model used.")
    query: str = Field(description="The original user query or instruction.")
    embedding: List[float] = Field(description="High-dimensional embedding vector representing the semantic content of the text query.")
    dimensions: int = Field(ge=1, description="Dimensionality of the text embedding vector.")
    task_type: TaskType = Field(description="Type of task being performed, which may influence how the text embedding is used in later stages.")
    key_entities: List[str] = Field(description="List of key entities or concepts extracted from the text query that may be relevant for multimodal processing.")
    processing_time: float = Field(description="Time taken to process the text query and generate the embedding, in seconds.")

# stage: projection interface
class ModalityAlignment(BaseModel):
    """Alignment metric between vision and text modalities."""
    metric: str = Field(description="Name of the alignment metric used (e.g., 'cosine_similarity_proxy').")
    score: float = Field(ge=-1.0, le=1.0, description="Alignment score between the visual features and text embedding, where higher scores indicate better alignment.")
    is_aligned: bool = Field(description="Boolean flag indicating whether the visual and textual representations are sufficiently aligned based on a predefined threshold.")
    alignment_quality: str = Field(description="Qualitative description of the alignment quality (e.g., 'strongly aligned', 'moderatly aligned', 'weakly aligned', 'misaligned').")

class ProjectionInterfaceResult(BaseModel):
    """Stage: Projection Interface output — shared latent space alignment."""
    stage: PipelineStage = Field(default=PipelineStage.PROJECTION_INTERFACE, description="Pipeline stage identifier.")
    visual_token_dim: int = Field(ge=1, description="Dimensionality of the tokenized visual features after projection.")
    text_token_dim: int = Field(ge=1, description="Dimensionality of the tokenized text embedding after projection.")
    projected_dim: int = Field(ge=1, description="Dimensionality of the shared latent space after projection.")
    alignment: ModalityAlignment = Field(description="Alignment metrics between the projected visual and textual representations.")
    visual_text_overlap: List[str] = Field(description="List of key concepts or entities that are present in both the visual and textual representations, indicating areas of strong multimodal overlap.")
    processing_time: float = Field(description="Time taken to perform the projection and alignment between modalities, in seconds.")

# stage: multimodal processor
class AttentionWeight(BaseModel):
    """Cross-attention weight between a text token and a visual region."""
    text_token: str = Field(description="The specific text token from the user query.")
    visual_region: str = Field(description="The specific visual region or feature that the text token is attending to.")
    weight: float = Field(ge=0.0, le=1.0, description="The attention weight indicating the strength of the connection between the text token and the visual region.")

class MultimodalProcessorResult(BaseModel):
    """Stage: Multimodal Processor — cross-attention fusion output."""
    stage: PipelineStage = Field(default=PipelineStage.MULTIMODAL_PROCESSOR, description="Pipeline stage identifier.")
    fusion_strategy: str = Field(description="Description of the fusion strategy used to combine visual and textual information (e.g., 'cross-attention', 'concatenation', 'gated fusion').")
    cross_attention: List[AttentionWeight] = Field(description="List of attention weights indicating how each text token attends to different visual regions, which can provide insights into the model's reasoning process.")
    grounded_concepts: List[str] = Field(description="List of concepts or entities that have been successfully grounded in both the visual and textual modalities, indicating a successful multimodal understanding.")
    multimodal_context: str = Field(description="A synthesized multimodal context that combines the visual features and text query into a coherent representation for the language model.")
    visual_text_conflicts: List[str] = Field(description="List of any detected conflicts or discrepancies between the visual and textual information (e.g., text mentions 'red car' but no red objects are detected in the image).")
    processing_time: float = Field(description="Time taken to perform the multimodal processing and fusion, in seconds.")

# stage: language model
class LanguageModelResult(BaseModel):
    """Stage: Language Model — vision-grounded generation output."""
    stage: PipelineStage = Field(default=PipelineStage.LANGUAGE_MODEL, description="Pipeline stage identifier.")
    model: str = Field(description="Name of the language model used for generation.")
    raw_response: str = Field(description="The raw text response generated by the language model before any post-processing.")
    finish_reason: str = Field(description="The reason why the language model finished generating (e.g., 'stop_token', 'max_tokens', 'end_turn').")
    prompt_tokens: int = Field(description="Number of tokens in the input prompt provided to the language model.")
    completion_tokens: int = Field(description="Number of tokens generated by the language model in the response.")
    total_tokens: int = Field(description="Total number of tokens consumed (prompt + completion), which can be used for cost estimation.")
    grounding_score: float = Field(ge=0.0, le=1.0, description="A score indicating how well the generated response is grounded in the visual information, which can be derived from the multimodal processor's attention weights and alignment metrics.")
    processing_time: float = Field(description="Time taken for the language model to generate the response, in seconds.")

# stage: output generation
class VLMOutput(BaseModel):
    """Final structured output of the full VLM pipeline."""
    request_id: str = Field(description="Unique ID for the request, matching the input request_id for traceability.")
    stage: PipelineStage = Field(default=PipelineStage.OUTPUT_GENERATION, description="Pipeline stage identifier.")
    status: ProcessingStatus = Field(default=ProcessingStatus.SUCCESS, description="Current processing status.")

    # stage payloads
    vision_encoder: VisionEncoderResult
    text_encoder: TextEncoderResult
    projection: ProjectionInterfaceResult
    multimodal_processor: MultimodalProcessorResult
    language_model: LanguageModelResult

    # final response
    response_text: str = Field(description="The final text response generated by the VLM pipeline, ready to be returned to the user.")
    visual_citations: List[str] = Field(description="List of visual elements (e.g., 'region_1: cat on sofa') that are explicitly cited in the final response, which can be used for explainability and traceability.")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score for the final response, which can be derived from the various alignment and grounding metrics throughout the pipeline.")
    total_pipeline_time: float = Field(description="Total time taken to process the request through the entire VLM pipeline, from input to final output, in seconds.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for logging, debugging, or future analysis.")

    @model_validator(mode="after")
    def populate_metadata(self) -> "VLMOutput":
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["alignment_score"] = self.projection.alignment.score
        self.metadata["total_tokens"] = self.language_model.total_tokens
        self.metadata["grounding_score"] = self.language_model.grounding_score
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
# IMAGE LOADER
# ==========================================
class ImageLoader:
    """
    Resolves ImagePayload from any source (file, URL, bytes, base64) into a validated (base64_string, mime_type) tuple ready for the API.
    """

    @staticmethod
    def load(image: ImagePayload) -> Tuple[str, str]:
        """
        Returns:
            (base64_encoded_string, mime_type)
        """
        if image.source_type == ImageSource.BASE64:
            # Already encoded — validate and return
            try:
                decoded = base64.b64decode(image.source)
                if len(decoded) > MAX_IMAGE_BYTES:
                    raise ValueError(
                        f"🚨Image size {len(decoded)} bytes exceeds limit {MAX_IMAGE_BYTES}."
                    )
            except Exception as e:
                raise ValueError(f"🚨 Invalid base64 image data: {e}")
            return image.source, image.mime_type

        if image.source_type == ImageSource.FILE:
            path = Path(image.source)
            if not path.exists():
                raise FileNotFoundError(f"🚨 Image file not found: {path}")
            raw_bytes = path.read_bytes()
            mime = mimetypes.guess_type(str(path))[0] or image.mime_type

        elif image.source_type == ImageSource.URL:
            try:
                with urlopen(image.source, timeout=15) as resp:
                    raw_bytes = resp.read()
                    content_type = resp.headers.get("Content-Type", image.mime_type)
                    mime = content_type.split(";")[0].strip()
            except URLError as e:
                raise RuntimeError(f"❌ Failed to fetch image from URL: {e}")

        elif image.source_type == ImageSource.BYTES:
            # source holds hex-encoded bytes for serialisability
            raw_bytes = bytes.fromhex(image.source)
            mime = image.mime_type

        else:
            raise ValueError(f"🚨 Unknown image source_type: {image.source_type}")

        if len(raw_bytes) > MAX_IMAGE_BYTES:
            raise ValueError(
                f"🚨 Image size {len(raw_bytes) / 1e6:.1f} MB exceeds limit {MAX_IMAGE_BYTES / 1e6:.0f} MB."
            )

        if mime not in SUPPORTED_MIME_TYPES:
            raise ValueError(
                f"🚨 Unsupported MIME type '{mime}'. "
                f"Supported: {SUPPORTED_MIME_TYPES}"
            )

        return base64.b64encode(raw_bytes).decode("utf-8"), mime

    @staticmethod
    def compute_sha256(b64_str: str) -> str:
        raw = base64.b64decode(b64_str)
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def estimate_patch_tokens(detail: str) -> int:
        """
        Estimate GPT-4.1 vision patch token count.
        detail=low → 85 tokens (fixed, regardless of image size)
        detail=high → up to 765 tokens (full 512px tile grid)
        """
        return 85 if detail == "low" else 765

# ==========================================
# PIPELINE STAGES
# ==========================================
class VisionEncoderStage:
    """
    Stage: VISION ENCODER
    Extracts a rich, structured visual feature map from the image.

    Architecture fidelity:
    - True VLM vision encoders (CLIP ViT, SigLIP) split the image into fixed-size patches (e.g. 14×14 px) and produce patch embeddings.
    - We replicate this with a GPT-4.1 structured vision pass that produces an equivalent semantic feature map: scene type, objects, regions, spatial layout, texture, lighting, depth — all structured and typed.
    - detail="high" activates GPT-4.1's full tile-based patch processing.
    """
    def run(self, b64_image: str, mime_type: str, detail: str, agent: BaseAIAgent) -> VisionEncoderResult:
        logger.info(
            f"⚙️ [VISION ENCODER] Extracting visual features (detail={detail})..."
        )
        t0 = time.perf_counter()
        data = agent._gpt_json_response(
            system=VISION_ENCODER_SYSTEM_PROMPT,
            user=[
                {
                    "type": "image_url",
                    "image_url": {
                        "url"   : f"data:{mime_type};base64,{b64_image}",
                        "detail": detail,
                    },
                },
                {
                    "type": "text",
                    "text": VISION_ENCODER_PROMPT_TEMPLATE,
                },
            ],
            max_tokens =1200,
            temperature=0.1,
        )
        feature_map = VisualFeatureMap(
            scene_type=data.get("scene_type", "unknown"),
            dominant_colors=data.get("dominant_colors", []),
            detected_objects=data.get("detected_objects", []),
            spatial_layout=data.get("spatial_layout", ""),
            visual_regions=[
                VisualRegion(**r) for r in data.get("visual_regions", [])
            ],
            texture_desc=data.get("texture_desc", ""),
            lighting_desc=data.get("lighting_desc", ""),
            estimated_depth=data.get("estimated_depth", "shallow"),
            contains_text=bool(data.get("contains_text", False)),
            contains_faces=bool(data.get("contains_faces", False)),
            visual_complexity=float(data.get("visual_complexity", 0.5)),
        )

        elapsed = time.perf_counter() - t0
        result = VisionEncoderResult(
            model=CHAT_MODEL,
            detail_level=detail,
            feature_map=feature_map,
            patch_token_est=ImageLoader.estimate_patch_tokens(detail),
            processing_time=elapsed
        )
        logger.info(
            f"✅ [VISION ENCODER] scene={feature_map.scene_type} | "
            f"objects={len(feature_map.detected_objects)} | "
            f"regions={len(feature_map.visual_regions)} | "
            f"complexity={feature_map.visual_complexity:.2f} | "
            f"time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [VISION ENCODER] feature map: {feature_map.model_dump_json(indent=2, ensure_ascii=False)}")
        return result

class TextEncoderStage:
    """
    Stage: TEXT ENCODER
    Encodes the text query into a semantic embedding vector and extracts key entities — the text-side representation for multimodal alignment.

    Architecture fidelity:
    - True VLMs (LLaVA, Flamingo, CLIP) use a shared text encoder (often CLIP's text tower) to produce text embeddings in the same space as the vision encoder's output.
    - We use text-embedding-3-small for the semantic vector and a GPT-4.1 NER pass for entity extraction.
    """
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, vlm_input: VLMInput, agent: BaseAIAgent) -> TextEncoderResult:
        logger.info(
            f"⚙️ [TEXT ENCODER] Encoding query: '{vlm_input.text_query[:60]}...'"
        )
        t0 = time.perf_counter()

        # sementic embedding
        embedding_response = agent._retry_api_call(
            self._client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=vlm_input.text_query
        )
        embedding = embedding_response.data[0].embedding

        # entity extraction
        entity_data = agent._gpt_json_response(
            system=(
                "You are a named entity extractor. Extract key entities from the query that are visually groundable. JSON only."
            ),
            user=(
                f"Query: {vlm_input.text_query}\n\n"
                "Extract entities that could appear in an image.\n"
                '{"key_entities": ["<entity1>", "<entity2>", ...]}'
            ),
            max_tokens =300,
            temperature=0.1,
        )
        key_entities = entity_data.get("key_entities", [])
        elapsed = time.perf_counter() - t0
        result = TextEncoderResult(
            model=EMBEDDING_MODEL,
            query=vlm_input.text_query,
            embedding=embedding,
            dimensions=len(embedding),
            task_type=vlm_input.task_type,
            key_entities=key_entities,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [TEXT ENCODER] dim={result.dimensions} | task_type={result.task_type} | "
            f"entities={key_entities} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [TEXT ENCODER] query={result.query} | embedding_vector: {embedding[:5]}...")
        return result

class ProjectionInterfaceStage:
    """
    Stage: PROJECTION INTERFACE
    Aligns vision and text representations into a shared latent space.

    Architecture fidelity:
    - In LLaVA, the projection interface is a 2-layer MLP that maps CLIP vision embeddings into the LLM's token embedding space.
    - In Flamingo, it's a Perceiver resampler that maps variable visual patch sequences to fixed-length visual tokens.
    - We replicate the alignment function by:
      (a) Computing a cosine-similarity proxy between visual concept vectors and text embedding direction.
      (b) Finding overlapping concepts between both modalities.
      (c) Producing alignment quality metrics used by the fusion stage.
    """
    def run(self, vision_result: VisionEncoderResult, text_result: TextEncoderResult, agent: BaseAIAgent) -> ProjectionInterfaceResult:
        logger.info(
            "⚙️ [PROJECTION INTERFACE] Aligning vision ↔ text modalities..."
        )
        t0 = time.perf_counter()

        # ── Cosine similarity proxy ────────────────────────────────────────
        # Represent visual content as a TF-IDF-style sparse vector over detected objects and scene terms; compare direction to text embedding
        visual_terms = (
            vision_result.feature_map.detected_objects
            + [vision_result.feature_map.scene_type]
            + vision_result.feature_map.dominant_colors
        )
        text_terms = text_result.key_entities + [text_result.task_type.value]

        # Convert to sets for overlap computation
        visual_set = {t.lower() for t in visual_terms}
        text_set = {t.lower() for t in text_terms}
        overlap = list(visual_set & text_set)

        # Jaccard similarity as alignment proxy
        union_size = len(visual_set | text_set)
        jaccard = len(overlap) / union_size if union_size > 0 else 0.0

        # Map Jaccard to [-1, 1] cosine-equivalent scale
        alignment_score = round(jaccard * 2 - 1.0, 4)   # 0 overlap → -1, full → +1
        alignment_score = max(-1.0, min(1.0, alignment_score))

        is_aligned = alignment_score >= PROJECTION_ALIGN_THRESHOLD - 1.0
        if alignment_score > 0.5:
            quality = "strong"
        elif alignment_score > 0.0:
            quality = "moderate"
        elif alignment_score > -0.5:
            quality = "weak"
        else:
            quality = "misaligned"

        if quality in ("weak", "misaligned"):
            logger.warning(
                f"⚠️ [PROJECTION] Alignment quality='{quality}' "
                f"(score={alignment_score:.4f}). Visual and text may be poorly matched."
            )
        
        # final visual-text overlap concepts for fusion stage
        overlap_data = agent._gpt_json_response(
            system=(
                "You are a multimodal alignment engine. Find concepts present in both visual content and text query. JSON only."
            ),
            user=(
                f"Visual scene: {vision_result.feature_map.scene_type}\n"
                f"Visual objects: {vision_result.feature_map.detected_objects}\n"
                f"Visual regions: "
                f"{[r.label for r in vision_result.feature_map.visual_regions]}\n\n"
                f"Text query: {text_result.query}\n"
                f"Text entities: {text_result.key_entities}\n\n"
                "List concepts grounded in BOTH modalities:\n"
                '{"visual_text_overlap": ["<concept1>", "<concept2>"]}'
            ),
            max_tokens=400,
            temperature=0.1,
        )
        visual_text_overlap = overlap_data.get("visual_text_overlap", overlap)

        elapsed = time.perf_counter() - t0
        result = ProjectionInterfaceResult(
            visual_token_dim=len(vision_result.feature_map.visual_regions),
            text_token_dim=text_result.dimensions,
            projected_dim=EMBEDDING_DIMENSIONS,
            alignment=ModalityAlignment(
                metric="jaccard_cosine_proxy",
                score=alignment_score,
                is_aligned=is_aligned,
                alignment_quality=quality
            ),
            visual_text_overlap=visual_text_overlap,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [PROJECTION] alignment={quality} | score={alignment_score:.4f} | "
            f"overlap={len(visual_text_overlap)} concepts | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [PROJECTION] overlap={visual_text_overlap}")
        return result

class MultimodalProcessorStage:
    """
    Stage: MULTIMODAL PROCESSOR
    Performs cross-modal attention fusion — grounding text tokens in specific visual regions and resolving conflicts between modalities.

    Architecture fidelity:
    - In Flamingo, gated cross-attention layers interleave visual and language tokens at each transformer layer.
    - In LLaVA-1.5, visual tokens are simply prepended to text tokens and processed by the LLM jointly.
    - We model cross-attention as explicit token-region attention weights computed by GPT-4.1, producing a grounded multimodal context.
    """
    def run(self, vision_result: VisionEncoderResult, text_result: TextEncoderResult, projection_result: ProjectionInterfaceResult, agent: BaseAIAgent) -> MultimodalProcessorResult:
        logger.info(
            "⚙️ [MULTIMODAL PROCESSOR] Computing cross-modal attention fusion..."
        )
        t0 = time.perf_counter()

        regions_desc = "\n".join(
            f" [{r.region_id}] {r.label} ({r.spatial_hint}): {r.description}"
            for r in vision_result.feature_map.visual_regions
        )

        data = agent._gpt_json_response(
            system=(
                "You are the multimodal fusion processor of a VLM. "
                "Compute cross-modal attention between text tokens and visual regions. "
                "Identify grounded concepts and conflicts. JSON only."
            ),
            user=MULTIMODAL_PROCESSOR_PROMPT_TEMPLATE.format(
                text_query=text_result.query,
                key_entities=text_result.key_entities,
                regions_desc=regions_desc,
                visual_text_overlap=json.dumps(projection_result.visual_text_overlap, ensure_ascii=False),
            ),
            max_tokens=1200,
            temperature=0.2,
        )
        cross_attn = [
            AttentionWeight(**a)
            for a in data.get("cross_attention", [])
        ]
        elapsed = time.perf_counter() - t0
        result = MultimodalProcessorResult(
            fusion_strategy="gated_cross_attention_simulation",
            cross_attention=cross_attn,
            grounded_concepts=data.get("grounded_concepts", []),
            multimodal_context=data.get("multimodal_context", ""),
            visual_text_conflicts=data.get("visual_text_conflicts", []),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [MULTIMODAL PROCESSOR] attention_pairs={len(cross_attn)} | grounded={len(result.grounded_concepts)} | "
            f"conflicts={len(result.visual_text_conflicts)} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [MULTIMODAL PROCESSOR] \ncross_attention={data.get('cross_attention', [])} |\ngrounded_concepts={result.grounded_concepts} |\nmultimodal_context={result.multimodal_context}")
        return result

class LanguageModelStage:
    """
    Stage: LANGUAGE MODEL
    The final generation stage — GPT-4.1 with image + text in a single
    multimodal call, enriched by all upstream stage context.

    This is the true VLM inference call: image pixels + fused context
    + original query → grounded natural language response.
    """
    def run(self, b64_image: str, mime_type: str, vlm_input: VLMInput, multimodal_processor: MultimodalProcessorResult, vision_result: VisionEncoderResult, agent: BaseAIAgent) -> LanguageModelResult:
        logger.info(f"⚙️ [LANGUAGE MODEL] Running vision-grounded generation (task={vlm_input.task_type.value})...")
        t0 = time.perf_counter()

        context_block = f"""
        [Multimodal Context from Fusion Stage]
        Grounded Concepts: {', '.join(multimodal_processor.grounded_concepts)}
        Context: {multimodal_processor.multimodal_context}
        Scene: {vision_result.feature_map.scene_type} | Objects: {', '.join(vision_result.feature_map.detected_objects[:10])}
        """
        if multimodal_processor.visual_text_conflicts:
            context_block += (
                f"\n⚠️ Conflicts detected: {', '.join(multimodal_processor.visual_text_conflicts)}"
            )
        response = agent._retry_api_call(
            agent.client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": LANGUAGE_MODEL_SYSTEM_PROMPT.format(task_type=vlm_input.task_type.value)},
                {
                    "role": "user", 
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{b64_image}",
                                "detail": vlm_input.detail_level,
                            },
                        },
                        {
                            "type": "text",
                            "text": vlm_input.text_query + context_block,
                        },
                    ],
                }
            ],
            temperature=vlm_input.temperature,
            max_tokens=vlm_input.max_tokens,
        )
        choice = response.choices[0]
        usage = response.usage
        raw_response = choice.message.content or ""
        elapsed = time.perf_counter() - t0

        # grounding score heuristic
        # count visual references in the response
        visual_terms = (
            vision_result.feature_map.detected_objects
            + vision_result.feature_map.dominant_colors
            + [r.label for r in vision_result.feature_map.visual_regions]
        )
        ref_count = sum(
            1 for term in visual_terms
            if term.lower() in raw_response.lower()
        )
        grounding_score = min(1.0, ref_count / max(len(visual_terms), 1))

        result = LanguageModelResult(
            model=CHAT_MODEL,
            raw_response=raw_response,
            finish_reason=choice.finish_reason or "unknown",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            grounding_score=round(grounding_score, 4),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [LANGUAGE MODEL] finish={result.finish_reason} | "
            f"tokens(p/c/t)={result.prompt_tokens}/{result.completion_tokens}/{result.total_tokens} | "
            f"grounding={grounding_score:.4f} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 [LANGUAGE MODEL] raw_response: {result.raw_response}")
        return result

# ==========================================
# VLM AGENT  —  Orchestrates all 8 pipeline stages
# ==========================================
class VLMAgent(BaseAIAgent):
    """Vision-Language Model (VLM) Agent.

    Pipeline:
    [Image Input] ‖ [Text Input] → 
    [Vision Encoder] ‖ [Text Encoder] (parallel modalities) →
    Projection Interface → Multimodal Processor → Language Model → Output.

    Core Principle: Dual-stream encoding → shared latent fusion → vision-grounded language generation.
    """
    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self._loader = ImageLoader()
        self._vision_encoder = VisionEncoderStage()
        self._text_encoder = TextEncoderStage(client)
        self._projection_interface = ProjectionInterfaceStage()
        self._multimodal_processor = MultimodalProcessorStage()
        self._language_model = LanguageModelStage()

    # public entry point
    def process(self, vlm_input: VLMInput) -> VLMOutput:
        """
        Execute the full VLM pipeline.

        Args:
            vlm_input: Validated VLMInput with image + text query.

        Returns:
            VLMOutput: Fully structured multimodal pipeline result.

        Raises:
            ValueError: On invalid input or image.
            RuntimeError: On unrecoverable API failure.
        """
        pipeline_start = time.perf_counter()
        logger.info(
            f"🚀 [VLM AGENT] Pipeline START | request_id={vlm_input.request_id}"
        )
        logger.info(
            f"📥 [INPUT] image_source={vlm_input.image.source_type.value} | "
            f"task={vlm_input.task_type.value} | "
            f"query='{vlm_input.text_query[:60]}...'"
        )

        try:
            # load and validate image
            logger.info("💬 [IMAGE LOADER] Resolving image input...")
            b64_image, mime_type = self._loader.load(vlm_input.image)
            sha256 = self._loader.compute_sha256(b64_image)
            logger.info(f"✅ [IMAGE INPUT] mime={mime_type} | b64_len={len(b64_image)} | sha256={sha256[:12]}...")

            # stage: vision encoder
            vision_result = checkpointer.load("VISION_ENCODER", VisionEncoderResult)
            if not vision_result:
                vision_result = self._vision_encoder.run(b64_image, mime_type, vlm_input.detail_level, self)

            # stage: text encoder (runs after vision to share entity context — could be parallel)
            text_result = checkpointer.load("TEXT_ENCODER", TextEncoderResult)
            if not text_result:
                text_result = self._text_encoder.run(vlm_input, self)
                checkpointer.save("TEXT_ENCODER", text_result)

            # stage: projection interface
            projection_result = checkpointer.load("PROJECTION_INTERFACE", ProjectionInterfaceResult)
            if not projection_result:
                projection_result = self._projection_interface.run(vision_result, text_result, self)
                checkpointer.save("PROJECTION_INTERFACE", projection_result)

            # stage: multimodal processor
            multimodal_processor_result = checkpointer.load("MULTIMODAL_PROCESSOR", MultimodalProcessorResult)
            if not multimodal_processor_result:
                multimodal_processor_result = self._multimodal_processor.run(vision_result, text_result, projection_result, self)
                checkpointer.save("MULTIMODAL_PROCESSOR", multimodal_processor_result)

            # stage: language model
            language_model_result = checkpointer.load("LANGUAGE_MODEL", LanguageModelResult)
            if not language_model_result:
                language_model_result = self._language_model.run(b64_image, mime_type, vlm_input, multimodal_processor_result, vision_result, self)
                checkpointer.save("LANGUAGE_MODEL", language_model_result)

            # stage: output generation
            # extract visual citations (regions referenced in response)
            visual_citations = [
                f"{r.region_id}: {r.label} ({r.spatial_hint})"
                for r in vision_result.feature_map.visual_regions
                if r.label.lower() in language_model_result.raw_response.lower()
            ]
            confidence = round((language_model_result.grounding_score + projection_result.alignment.score + 1.0) / 3.0, 4)
            total_time = time.perf_counter() - pipeline_start
            output = VLMOutput(
                request_id=vlm_input.request_id,
                vision_encoder=vision_result,
                text_encoder=text_result,
                projection=projection_result,
                multimodal_processor=multimodal_processor_result,
                language_model=language_model_result,
                response_text=language_model_result.raw_response,
                visual_citations=visual_citations,
                confidence=confidence,
                total_pipeline_time=round(total_time, 4),
                metadata={
                    **vlm_input.metadata,
                    "model_vision": CHAT_MODEL,
                    "model_embedding": EMBEDDING_MODEL,
                    "image_sha256": sha256,
                    "detail_level": vlm_input.detail_level,
                },
            )
            self.logger.info(
                f"🎉 [VLM AGENT] Pipeline COMPLETE | total_time={total_time:.4f}s | "
                f"tokens={language_model_result.total_tokens} | grounding={language_model_result.grounding_score:.4f} | "
                f"alignment={projection_result.alignment.alignment_quality} | citations={len(visual_citations)}"
            )
            logger.debug(f"🔍 [VLM AGENT] Final output: {output.model_dump_json(indent=2, ensure_ascii=False)}")
            return output
        except Exception as e:
            elapsed = time.perf_counter() - pipeline_start
            self.logger.error(
                f"❌ [VLM AGENT] Pipeline FAILED after {elapsed:.4f}s | "
                f"error={type(e).__name__}: {e}"
            )
            raise RuntimeError(f"VLM pipeline failed: {e}") from e
        
    @staticmethod
    def from_file(
        image_path: str,
        text_query: str,
        task_type: TaskType = TaskType.VISUAL_QA,
        **kwargs,
    ) -> VLMInput:
        """Build VLMInput from a local image file path."""
        p = Path(image_path)
        mime = mimetypes.guess_type(str(p))[0] or "image/jpeg"
        return VLMInput(
            image=ImagePayload(
                source_type=ImageSource.FILE,
                source=str(p),
                mime_type=mime,
            ),
            text_query=text_query,
            task_type=task_type,
            **kwargs,
        )

    @staticmethod
    def from_url(
        image_url: str,
        text_query: str,
        task_type: TaskType = TaskType.VISUAL_QA,
        mime_type: str = "image/jpeg",
        **kwargs,
    ) -> VLMInput:
        """Build VLMInput from an image URL."""
        return VLMInput(
            image=ImagePayload(
                source_type=ImageSource.URL,
                source=image_url,
                mime_type=mime_type,
            ),
            text_query=text_query,
            task_type=task_type,
            **kwargs,
        )

    @staticmethod
    def from_base64(
        b64_string: str,
        text_query: str,
        mime_type: str = "image/jpeg",
        task_type: TaskType = TaskType.VISUAL_QA,
        **kwargs,
    ) -> VLMInput:
        """Build VLMInput from a pre-encoded base64 image string."""
        return VLMInput(
            image=ImagePayload(
                source_type=ImageSource.BASE64,
                source=b64_string,
                mime_type=mime_type,
            ),
            text_query=text_query,
            task_type=task_type,
            **kwargs,
        )
    
    # display helper
    def display_output(self, output: VLMOutput) -> None:
        div = "=" * 80
        print(f"\n{div}")
        print(" 🟢 Vision-Language Model (VLM) Pipeline Result")
        print(f"{div}")
        print(f"Request ID: {output.request_id}")
        print(f"Status: {output.status.value}")
        print(f"Confidence: {output.confidence:.4f}")
        print(f"Total Time: {output.total_pipeline_time}s")

        print(f"{div}")
        print(f"📤 OUTPUT GENERATION\n")
        if output.visual_citations:
            print(f"Visual Citations : {output.visual_citations}\n")
        print(f"{output.response_text}")
        print(f"\n{div}\n")

# ==========================================
# Instatiation
# ==========================================
def create_vlm_agent(api_key: str = TOKEN, endpoint: str = ENDPOINT) -> VLMAgent:
    """Factory function to create an instance of VLMAgent with the provided API credentials."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    logger.info("🏭 [FACTORY] VLMAgent instantiated and ready.")
    return VLMAgent(client)

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    agent = create_vlm_agent()

    # ═══════════════════════════════════════════════════════════════════════
    # USAGE PATTERN 1: From URL
    # ═══════════════════════════════════════════════════════════════════════
    # vlm_input = VLMAgent.from_url(
    #     image_url  = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
    #     text_query = (
    #         "Describe what you see in this image in detail. "
    #         "Identify all objects, their spatial relationships, "
    #         "colours, and any text present."
    #     ),
    #     task_type  = TaskType.SCENE_UNDERSTANDING,
    #     mime_type  = "image/png",
    #     metadata   = {"source": "vlm_agent_demo_url", "version": "1.0"},
    # )
    # result = agent.process(vlm_input)
    # agent.display_output(result)

    # ═══════════════════════════════════════════════════════════════════════
    # USAGE PATTERN 2: From local file
    # Uncomment and set path to test with a local image
    # ═══════════════════════════════════════════════════════════════════════
    vlm_input = VLMAgent.from_file(
        image_path="assets/image/llm.png",
        text_query="What is happening in this image?",
        task_type=TaskType.VISUAL_QA,
        metadata={"source": "vlm_agent_demo_file", "version": "1.0"},
    )
    result = agent.process(vlm_input)
    agent.display_output(result)

    # ═══════════════════════════════════════════════════════════════════════
    # USAGE PATTERN 3: From base64 string
    # Uncomment to test with a pre-encoded image
    # ═══════════════════════════════════════════════════════════════════════
    # with open("/path/to/image.png", "rb") as f:
    #     b64 = base64.b64encode(f.read()).decode()
    # vlm_input = VLMAgent.from_base64(
    #     b64_string = b64,
    #     text_query = "Extract all text visible in this image.",
    #     task_type  = TaskType.TEXT_EXTRACTION,
    #     mime_type  = "image/png",
    #     metadata   = {"source": "vlm_agent_demo_base64", "version": "1.0"},
    # )
    # result = agent.process(vlm_input)
    # agent.display_output(result)
