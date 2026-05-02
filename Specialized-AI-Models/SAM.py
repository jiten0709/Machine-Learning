import os, uuid, json, math, random, time, numpy as np, tiktoken, re, base64, hashlib, mimetypes
from openai import OpenAI, APIError, RateLimitError, APITimeoutError, APIConnectionError
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Set
from pydantic import BaseModel, Field, field_validator, model_validator
from datetime import datetime, timezone
from pathlib import Path
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.request import urlopen
from urllib.error import URLError

from dotenv import load_dotenv
load_dotenv()

from utils.logging_setup import get_logger
logger = get_logger(__name__, log_file="sam.log")

# ==========================================
# Variable Configuration
# ==========================================
from utils.state_checkpointer import StateCheckpointer
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# FOR DEMO-1
# checkpointer = StateCheckpointer(
#     directory=CHECKPOINT_DIR, 
#     filename="sam_checkpoint_demo_1.json",
#     logger=logger
# )

# FOR DEMO-2
# checkpointer = StateCheckpointer(
#     directory=CHECKPOINT_DIR, 
#     filename="sam_checkpoint_demo_2.json",
#     logger=logger
# )

# FOR DEMO-3
checkpointer = StateCheckpointer(
    directory=CHECKPOINT_DIR, 
    filename="sam_checkpoint_demo_3.json",
    logger=logger
)

TOKEN = os.environ['GITHUB_TOKEN']
ENDPOINT = os.environ['GITHUB_ENDPOINT']
CHAT_MODEL = os.environ['GITHUB_MODEL_NAME']
EMBEDDING_MODEL = os.environ['GITHUB_EMBED_NAME']
EMBEDDING_DIMENSIONS = 1536

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0

# SAM image encoder constants (ViT-H patch grid simulation)
IMAGE_ENCODER_PATCH_SIZE = 16          # 16×16 patch grid
IMAGE_ENCODER_EMBED_DIM  = 256         # patch embedding dimension (SAM uses 256)
PATCH_GRID_H = 16          # 16 rows
PATCH_GRID_W = 16          # 16 cols
TOTAL_PATCHES = PATCH_GRID_H * PATCH_GRID_W   # 256 patches

# Mask decoder constants
NUM_MASK_CANDIDATES = 3           # SAM always predicts 3 mask candidates
MASK_GRID_SIZE = 16          # low-res mask grid (16×16 → upsampled)
IOU_THRESHOLD = 0.5         # minimum IoU for valid mask
STABILITY_THRESHOLD = 0.6         # minimum stability score for mask selection

# Feature correlation
CORRELATION_TOP_K = 5           # top-k correlated patches per prompt token

# Image constraints
MAX_IMAGE_BYTES = 20 * 1024 * 1024
SUPPORTED_MIMES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
VISION_DETAIL = "high"

# ==========================================
# ENUMS
# ==========================================
class PipelineStage(str, Enum):
    PROMPT_INPUT = "PROMPT_INPUT"
    IMAGE_INPUT = "IMAGE_INPUT"
    PROMPT_ENCODER = "PROMPT_ENCODER"
    IMAGE_ENCODER = "IMAGE_ENCODER"
    IMAGE_EMBEDDING = "IMAGE_EMBEDDING"
    MASK_DECODER = "MASK_DECODER"
    FEATURE_CORRELATION = "FEATURE_CORRELATION"
    SEGMENTATION_OUTPUT = "SEGMENTATION_OUTPUT"

class PromptType(str, Enum):
    POINT = "POINT"        # (x, y) coordinate + foreground/background label
    BOX = "BOX"          # (x1, y1, x2, y2) bounding box
    TEXT = "TEXT"         # natural language description
    MASK = "MASK"         # coarse binary mask hint
    EVERYTHING = "EVERYTHING"   # segment all objects (no prompt)

class PointLabel(str, Enum):
    FOREGROUND = "FOREGROUND"   # 1 — click on the object
    BACKGROUND = "BACKGROUND"   # 0 — click away from object

class ImageSource(str, Enum):
    FILE = "FILE"
    URL = "URL"
    BASE64 = "BASE64"

class ProcessingStatus(str, Enum):
    PENDING = "PENDING"
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"   # some masks below quality threshold
    FAILED = "FAILED"

# ==========================================
# PYDANTIC MODELS
# ==========================================
# Prompt sub-types 

class PointPrompt(BaseModel):
    """A single (x, y) coordinate prompt with foreground/background label."""
    x: float = Field(ge=0.0, le=1.0, description="Normalised x (0–1)")
    y: float = Field(ge=0.0, le=1.0, description="Normalised y (0–1)")
    label: PointLabel = Field(default=PointLabel.FOREGROUND, description="Foreground (1) or background (0) point")

class BoxPrompt(BaseModel):
    """Bounding box prompt (x1, y1, x2, y2) in normalised [0,1] coords."""
    x1:float = Field(ge=0.0, le=1.0, description="Normalised x1 (0–1, left)")
    y1:float = Field(ge=0.0, le=1.0, description="Normalised y1 (0–1, top)")
    x2:float = Field(ge=0.0, le=1.0, description="Normalised x2 (0–1, right)")
    y2:float = Field(ge=0.0, le=1.0, description="Normalised y2 (0–1, bottom)")

    @model_validator(mode="after")
    def validate_box(self) -> "BoxPrompt":
        if self.x1 >= self.x2 or self.y1 >= self.y2:
            raise ValueError("🚨 Box must have x1<x2 and y1<y2.")
        return self

class TextPrompt(BaseModel):
    """Natural language description of the object to segment."""
    description: str = Field(..., min_length=2, max_length=512, description="Describe the object to segment (2–512 chars).")

class MaskPrompt(BaseModel):
    """Coarse binary mask hint (16×16 grid, values 0.0–1.0)."""
    mask_grid: List[List[float]] = Field(description=f"16×16 grid of float values (0=background, 1=foreground)")

    @field_validator("mask_grid")
    @classmethod
    def validate_grid(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) != MASK_GRID_SIZE:
            raise ValueError(f"🚨 Mask grid must have {MASK_GRID_SIZE} rows.")
        for row in v:
            if len(row) != MASK_GRID_SIZE:
                raise ValueError(f"🚨 Each row must have {MASK_GRID_SIZE} cols.")
        return v

# Main input 

class ImagePayload(BaseModel):
    """Validated image payload."""
    source_type: ImageSource = Field(..., description="How the image is provided: FILE, URL, or BASE64.")
    source: str = Field(..., description="The image source: file path, URL, or base64 string.")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the image (e.g. image/jpeg).")

    @field_validator("mime_type")
    @classmethod
    def validate_mime(cls, v: str) -> str:
        if v not in SUPPORTED_MIMES:
            raise ValueError(f"🚨 Unsupported MIME: {v}. Supported: {SUPPORTED_MIMES}")
        return v

class SAMInput(BaseModel):
    """Combined prompt + image input to the SAM pipeline."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique ID for this segmentation request.")
    image: ImagePayload = Field(..., description="Input image payload with source and MIME type.")
    prompt_type: PromptType = Field(default=PromptType.TEXT, description="Type of prompt provided (POINT, BOX, TEXT, MASK, EVERYTHING).") 

    # Prompt payloads (only one should be set per prompt_type)
    point_prompts: List[PointPrompt] = Field(default_factory=list, description="List of point prompts (required if prompt_type=POINT).")
    box_prompt: Optional[BoxPrompt] = Field(default=None, description="Box prompt (required if prompt_type=BOX).")
    text_prompt: Optional[TextPrompt] = Field(default=None, description="Text prompt (required if prompt_type=TEXT).")
    mask_prompt: Optional[MaskPrompt] = Field(default=None, description="Mask prompt (required if prompt_type=MASK).")

    multimask_output: bool = Field(default=True, description="Whether to return multiple mask candidates (SAM's default is True).")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_prompt_consistency(self) -> "SAMInput":
        """Ensure the right prompt payload is provided for the prompt type."""
        if self.prompt_type == PromptType.POINT and not self.point_prompts:
            raise ValueError("🚨 POINT prompt_type requires at least one point_prompt.")
        if self.prompt_type == PromptType.BOX and self.box_prompt is None:
            raise ValueError("🚨 BOX prompt_type requires box_prompt.")
        if self.prompt_type == PromptType.TEXT and self.text_prompt is None:
            raise ValueError("🚨 TEXT prompt_type requires text_prompt.")
        if self.prompt_type == PromptType.MASK and self.mask_prompt is None:
            raise ValueError("🚨 MASK prompt_type requires mask_prompt.")
        self.metadata["created_at"]   = datetime.now(timezone.utc).isoformat()
        self.metadata["prompt_type"]  = self.prompt_type.value
        return self

# Stage: Prompt Encoder 

class SparsePromptToken(BaseModel):
    """Encoded sparse prompt token (point or box corner)."""
    token_id: str = Field(..., description="Unique token ID (e.g. PT001, BX_TL).")
    token_type: str = Field(..., description="Type of token (e.g. point_fg, point_bg, box_tl, box_br).")
    x: float = Field(description="Normalised x coordinate (0–1).")
    y: float = Field(description="Normalised y coordinate (0–1).")
    embedding: List[float] = Field(..., description="Positional embedding vector (e.g. 64-d).")
    description: str = Field(..., description="Human-readable description of this token.")

class DensePromptFeatures(BaseModel):
    """Encoded dense prompt features (text or mask)."""
    source_type: str = Field(..., description="Source of dense features (e.g. 'text' or 'mask').")
    embedding: List[float] = Field(..., description="Dense embedding vector (e.g. 1536-d for text, 256-d for mask).")
    semantic_tags: List[str] = Field(default_factory=list, description="Extracted semantic tags from the prompt (for text).")
    density_map: List[float] = Field(default_factory=list, description="Flattened density map (e.g. 256-d for 16×16 grid)")

class PromptEncoderResult(BaseModel):
    """Stage: Prompt Encoder output."""
    stage: PipelineStage = PipelineStage.PROMPT_ENCODER
    prompt_type: PromptType = Field(..., description="Type of prompt encoded.")
    sparse_tokens: List[SparsePromptToken] = Field(default_factory=list, description="List of encoded sparse prompt tokens.")
    dense_features: Optional[DensePromptFeatures] = Field(default=None, description="Encoded dense prompt features (if applicable).")
    prompt_summary: str = Field(..., description="Human-readable summary of the encoded prompt.")
    prompt_embedding: List[float] = Field(..., description="Unified prompt embedding vector (e.g. 1536-d) to be fused with image features.")
    processing_time: float = Field(..., description="Time taken to encode the prompt (in seconds).")

# Stage: Image Encoder 

class ImagePatch(BaseModel):
    """A single ViT-style image patch with its feature embedding."""
    patch_id: str = Field(..., description="Unique patch ID (e.g. P0001).")
    row: int = Field(..., description="Row index of the patch in the grid.")
    col: int = Field(..., description="Column index of the patch in the grid.")
    x_norm: float = Field(..., description="Normalised x coordinate of patch center (0–1).")
    y_norm: float = Field(..., description="Normalised y coordinate of patch center (0–1).")
    description: str = Field(..., description="Human-readable description of the patch location.")
    feature: List[float] = Field(..., description="Feature embedding vector for this patch (e.g. 256-d).")
    salience: float = Field(ge=0.0, le=1.0, description="Estimated salience of this patch for segmentation (0–1).")
    object_label: str = Field(..., description="Semantic label of the dominant object in this patch.")

class ImageEncoderResult(BaseModel):
    """Stage: Image Encoder output."""
    stage: PipelineStage = PipelineStage.IMAGE_ENCODER
    model: str = Field(..., description="Name of the image encoder model used (e.g. 'ViT-H').")
    patch_grid_h: int = Field(default=PATCH_GRID_H, description="Number of patch rows in the grid.")
    patch_grid_w: int = Field(default=PATCH_GRID_W, description="Number of patch columns in the grid.")
    total_patches: int = Field(default=TOTAL_PATCHES, description="Total number of patches (rows × cols).")
    patches: List[ImagePatch] = Field(default_factory=list, description="List of encoded image patches with features and metadata.")
    scene_summary: str = Field(..., description="Human-readable summary of the scene based on patch features.")
    dominant_objects: List[str] = Field(default_factory=list, description="List of dominant object labels detected across patches.")
    processing_time: float = Field(..., description="Time taken to encode the image (in seconds).")

# Stage: Image Embedding 
class SpatialAttentionCell(BaseModel):
    """One cell in the spatial attention map (prompt → image)."""
    row: int = Field(..., description="Row index of the attention cell.")
    col: int = Field(..., description="Column index of the attention cell.")
    attention: float = Field(ge=0.0, le=1.0, description="Attention weight for this cell (0–1).")

class ImageEmbeddingResult(BaseModel):
    """Stage: Image Embedding — fused dense image representation."""
    stage: PipelineStage = PipelineStage.IMAGE_EMBEDDING
    global_embedding: List[float] = Field(..., description="Global image embedding vector (e.g. 1536-d) after fusion with prompt.")
    patch_embedding_dim: int = Field(default=IMAGE_ENCODER_EMBED_DIM, description="Dimension of patch embeddings (e.g. 256-d).")
    pooled_patch_vec: List[float] = Field(..., description="Pooled patch embedding vector (e.g. mean of all patch features).")
    spatial_attention: List[SpatialAttentionCell] = Field(default_factory=list, description="Spatial attention map cells indicating prompt focus areas.")
    high_salience_patches: List[str] = Field(default_factory=list, description="IDs of patches with salience above a certain threshold.")
    processing_time: float = Field(..., description="Time taken to compute the image embedding (in seconds).")

# Stage: Mask Decoder

class MaskRegion(BaseModel):
    """A semantically described region within a predicted mask."""
    region_id: str = Field(..., description="Unique region ID (e.g. R001).")
    description: str = Field(..., description="Human-readable description of this region (e.g. 'top-left corner').")
    spatial_bounds: str = Field(..., description="Approximate spatial bounds of this region in normalised coordinates (e.g. 'x1=0.0,y1=0.0,x2=0.5,y2=0.5').")
    coverage_ratio: float = Field(ge=0.0, le=1.0, description="Fraction of the mask area covered by this region (0–1).")

class SegmentationMask(BaseModel):
    """A single predicted segmentation mask candidate."""
    mask_id: str = Field(..., description="Unique mask ID (e.g. M001).")
    rank: int = Field(..., description="Rank of this mask candidate (1=best, 2=second-best, etc.).")
    iou_estimate: float = Field(ge=0.0, le=1.0, description="Estimated IoU of this mask with the true object (0–1).")
    stability_score: float = Field(ge=0.0, le=1.0, description="Stability score of this mask across SAM's internal augmentations (0–1).")
    confidence: float = Field(ge=0.0, le=1.0, description="Overall confidence score for this mask candidate (0–1).")

    # Mask representation
    rle_encoding: str = Field(..., description="Run-Length Encoding (RLE) string representing the binary mask grid.")
    mask_grid: List[List[int]] = Field(default_factory=list, description="Decoded 16×16 binary mask grid (values 0 or 1).")
    regions: List[MaskRegion] = Field(default_factory=list, description="Semantically described regions within this mask.")

    # Mask properties
    area_ratio: float = Field(ge=0.0, le=1.0, description="Area of the mask as a fraction of the total image area (0–1).")
    object_class: str = Field(..., description="Semantic class label of the object segmented by this mask (e.g. 'cat', 'car').")
    object_description: str = Field(..., description="Human-readable description of the segmented object based on mask features and regions.")
    is_valid: bool = Field(..., description="Whether this mask is considered valid based on IoU and stability thresholds.")

class MaskDecoderResult(BaseModel):
    """Stage: Mask Decoder output."""
    stage: PipelineStage = PipelineStage.MASK_DECODER
    masks: List[SegmentationMask] = Field(default_factory=list, description="List of predicted segmentation mask candidates with metadata.")
    best_mask_id: str = Field(description="ID of the best mask candidate selected based on confidence and validity.")
    num_valid_masks: int = Field(..., description="Number of mask candidates that passed the validity criteria.")
    decoding_strategy: str = Field(..., description="Description of the decoding strategy used (e.g. 'multimask_output=True with IoU and stability filtering').")
    processing_time: float = Field(..., description="Time taken to decode the masks (in seconds).")

# Stage: Feature Correlation 

class PatchCorrelation(BaseModel):
    """Correlation score between a prompt token and an image patch."""
    prompt_token_id: str = Field(..., description="ID of the prompt token (e.g. PT001, BX_TL).")
    patch_id: str = Field(..., description="ID of the image patch (e.g. P0001).")
    correlation: float = Field(ge=-1.0, le=1.0, description="Correlation score between the prompt token and patch feature (-1 to 1).")
    row: int = Field(..., description="Row index of the correlated patch.")
    col: int = Field(..., description="Column index of the correlated patch.")

class FeatureCorrelationResult(BaseModel):
    """Stage: Feature Correlation output."""
    stage: PipelineStage = PipelineStage.FEATURE_CORRELATION
    correlations: List[PatchCorrelation] = Field(default_factory=list, description="List of correlation scores between prompt tokens and image patches.")
    top_k_patches: List[str] = Field(default_factory=list, description="IDs of the top-k most correlated patches across all prompt tokens.")
    mean_correlation: float = Field(..., description="Mean correlation score across all prompt token-patch pairs.")
    max_correlation: float = Field(..., description="Maximum correlation score observed between any prompt token and patch.")
    prompt_patch_alignment: float = Field(ge=0.0, le=1.0, description="Overall alignment score between the prompt and image features based on correlation patterns (0–1).")
    processing_time: float = Field(..., description="Time taken to compute feature correlations (in seconds).")

# Final Output 

class SAMOutput(BaseModel):
    """Final structured output of the full SAM pipeline."""
    request_id: str = Field(..., description="Unique ID for this segmentation request (carried over from input).")
    stage: PipelineStage = PipelineStage.SEGMENTATION_OUTPUT
    status: ProcessingStatus = Field(default=ProcessingStatus.SUCCESS, description="Overall processing status of the segmentation request (PENDING, SUCCESS, PARTIAL, FAILED).")

    # Stage payloads
    prompt_encoder: PromptEncoderResult 
    image_encoder: ImageEncoderResult
    image_embedding: ImageEmbeddingResult
    mask_decoder: MaskDecoderResult
    feature_correlation: FeatureCorrelationResult

    # Final segmentation result
    best_mask: SegmentationMask = Field(..., description="The best segmentation mask candidate selected based on validity and confidence.")
    segmentation_summary: str = Field(..., description="Human-readable summary of the segmentation result and the identified object.")
    total_pipeline_time: float = Field(..., description="Total time taken to process the entire SAM pipeline (in seconds).")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the processing (e.g. timestamps, quality metrics).")

    @model_validator(mode="after")
    def populate_metadata(self) -> "SAMOutput":
        self.metadata["completed_at"] = datetime.now(timezone.utc).isoformat()
        self.metadata["best_mask_iou"] = self.best_mask.iou_estimate
        self.metadata["num_valid_masks"] = self.mask_decoder.num_valid_masks
        self.metadata["prompt_alignment"] = self.feature_correlation.prompt_patch_alignment
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
# IMAGE LOADER
# ==========================================
class ImageLoader:
    """Resolves ImagePayload → (base64_string, mime_type, sha256)."""

    @staticmethod
    def load(image: ImagePayload) -> Tuple[str, str, str]:
        if image.source_type == ImageSource.BASE64:
            b64 = image.source
            raw = base64.b64decode(b64)
        elif image.source_type == ImageSource.FILE:
            path = Path(image.source)
            if not path.exists():
                raise FileNotFoundError(f"🚨 Image not found: {path}")
            raw = path.read_bytes()
            mime = mimetypes.guess_type(str(path))[0] or image.mime_type
            image.mime_type = mime
            b64  = base64.b64encode(raw).decode()
        elif image.source_type == ImageSource.URL:
            try:
                with urlopen(image.source, timeout=15) as r:
                    raw = r.read()
                    ct = r.headers.get("Content-Type", image.mime_type)
                    image.mime_type = ct.split(";")[0].strip()
            except URLError as e:
                raise RuntimeError(f"🚨 Failed to fetch image: {e}")
            b64 = base64.b64encode(raw).decode()
        else:
            raise ValueError(f"🚨 Unknown source_type: {image.source_type}")

        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError(f"🚨 Image too large: {len(raw)/1e6:.1f}MB > 20MB limit")
        if image.mime_type not in SUPPORTED_MIMES:
            raise ValueError(f"🚨 Unsupported MIME: {image.mime_type}")

        sha256 = hashlib.sha256(raw).hexdigest()
        return b64, image.mime_type, sha256
    
# ==========================================
# HELPER FUNCTIONS
# ==========================================
def positional_embedding_2d(x: float, y: float, dim: int = 64) -> List[float]:
    """
    2D sinusoidal positional embedding for (x, y) coordinates.
    Used to encode sparse point/box prompts into the embedding space.
    First dim//2 dims encode x, second dim//2 encode y.
    """
    half = dim // 2
    enc = []
    for i in range(half // 2):
        denom = 10_000 ** (2 * i / half)
        enc.append(math.sin(x * math.pi * 2 / denom))
        enc.append(math.cos(x * math.pi * 2 / denom))
    for i in range(half // 2):
        denom = 10_000 ** (2 * i / half)
        enc.append(math.sin(y * math.pi * 2 / denom))
        enc.append(math.cos(y * math.pi * 2 / denom))
    return enc[:dim]

def rle_encode(mask_grid: List[List[int]]) -> str:
    """
    Run-Length Encode a binary mask grid → COCO-style RLE string.
    Format: "count1,count2,count3,..." alternating 0s and 1s.
    """
    flat = [cell for row in mask_grid for cell in row]
    if not flat:
        return ""
    runs: List[int] = []
    current = flat[0]
    count = 1
    # Start RLE from 0 (background), so if first pixel is 1, prepend 0
    if current == 1:
        runs.append(0)
    for val in flat[1:]:
        if val == current:
            count += 1
        else:
            runs.append(count)
            count = 1
            current = val
    runs.append(count)
    return ",".join(map(str, runs))

def rle_decode(rle: str, h: int = MASK_GRID_SIZE, w: int = MASK_GRID_SIZE) -> List[List[int]]:
    """Decode COCO-style RLE string back to binary mask grid."""
    if not rle:
        return [[0] * w for _ in range(h)]
    counts = list(map(int, rle.split(",")))
    flat = []
    val = 0
    for c in counts:
        flat.extend([val] * c)
        val = 1 - val
    flat = flat[:h * w] + [0] * max(0, h * w - len(flat))
    return [flat[r * w:(r + 1) * w] for r in range(h)]

def cosine_sim(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=np.float64)
    vb = np.array(b, dtype=np.float64)
    den = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / den) if den > 0 else 0.0


def generate_mask_grid(regions: List[MaskRegion], scene_w: int = MASK_GRID_SIZE, scene_h: int = MASK_GRID_SIZE) -> List[List[int]]:
    """
    Generate a 16×16 binary mask grid from described spatial regions.
    Each region specifies approximate bounding box in normalised coords.
    """
    grid = [[0] * scene_w for _ in range(scene_h)]
    for region in regions:
        x1 = int(region.get("x1", 0.0) * scene_w)
        y1 = int(region.get("y1", 0.0) * scene_h)
        x2 = int(region.get("x2", 1.0) * scene_w)
        y2 = int(region.get("y2", 1.0) * scene_h)
        for r in range(max(0, y1), min(scene_h, y2)):
            for c in range(max(0, x1), min(scene_w, x2)):
                grid[r][c] = 1
    return grid

# ==========================================
# PIPELINE STAGES
# ==========================================
class PromptEncoderStage:
    """
    Stage: PROMPT ENCODER
    Encodes any combination of SAM's supported prompt types into
    a unified embedding representation.

    SAM distinguishes two prompt classes:
    ┌──────────────────────────────────────────────────────────┐
    │ SPARSE prompts: points, boxes                            │
    │   → Encoded as positional embeddings + learned type bias │
    │   → Each point/corner = one token in sparse token list   │
    │                                                          │
    │ DENSE prompts: text, coarse masks                        │
    │   → Encoded as full feature maps over the image space    │
    │   → Added element-wise to the image embedding            │
    └──────────────────────────────────────────────────────────┘

    We encode:
    - Points/boxes → 2D sinusoidal positional embeddings (64-d)
    - Text → OpenAI text embedding (1536-d) + semantic tag extraction
    - Mask → spatial density map (256-d flattened)
    - Unified → single 1536-d prompt representation via API
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, sam_input: SAMInput, agent: BaseAIAgent) -> PromptEncoderResult:
        logger.info(f"⚙️ [PROMPT ENCODER] Encoding {sam_input.prompt_type.value} prompt...")
        t0 = time.perf_counter()

        sparse_tokens  : List[SparsePromptToken] = []
        dense_features : Optional[DensePromptFeatures] = None

        # Sparse: Points 
        for i, pt in enumerate(sam_input.point_prompts):
            emb = positional_embedding_2d(pt.x, pt.y, dim=64)
            token_type = (
                "point_fg" if pt.label == PointLabel.FOREGROUND
                else "point_bg"
            )
            # Add label bias: foreground tokens get positive offset
            bias = 0.1 if pt.label == PointLabel.FOREGROUND else -0.1
            emb = [v + bias for v in emb]
            sparse_tokens.append(SparsePromptToken(
                token_id=f"PT_{i:03d}",
                token_type=token_type,
                x=pt.x, 
                y=pt.y,
                embedding=emb,
                description=f"{'Foreground' if pt.label == PointLabel.FOREGROUND else 'Background'} point at ({pt.x:.2f}, {pt.y:.2f})",
            ))
            logger.debug(f"📍 Point[{i}] ({pt.x:.2f},{pt.y:.2f}) [{pt.label.value}]")

        # Sparse: Box 
        if sam_input.box_prompt:
            bp = sam_input.box_prompt
            # Encode top-left and bottom-right corners as two tokens
            for corner, (cx, cy), label in [
                ("box_tl", (bp.x1, bp.y1), "Top-left"),
                ("box_br", (bp.x2, bp.y2), "Bottom-right"),
            ]:
                emb = positional_embedding_2d(cx, cy, dim=64)
                sparse_tokens.append(SparsePromptToken(
                    token_id=f"BX_{corner.upper()}",
                    token_type=corner,
                    x=cx,
                    y=cy,
                    embedding=emb,
                    description=f"{label} corner of bounding box at ({cx:.2f}, {cy:.2f})",
                ))
            logger.debug(
                f"📦 Box [{bp.x1:.2f},{bp.y1:.2f}→{bp.x2:.2f},{bp.y2:.2f}]"
            )

        # Dense: Text 
        if sam_input.text_prompt:
            tp  = sam_input.text_prompt
            emb_resp = agent._retry_api_call(
                self._client.embeddings.create,
                model = EMBEDDING_MODEL,
                input = tp.description,
            )
            text_emb = emb_resp.data[0].embedding

            # Extract semantic tags from text prompt
            tag_data = agent._gpt_json_response(
                system=(
                    "You are a semantic tag extractor for image segmentation. "
                    "Extract visual attributes of the object to segment. JSON only."
                ),
                user=(
                    f"Segmentation prompt: '{tp.description}'\n\n"
                    "Extract semantic tags and generate a uniform density map.\n"
                    '{"semantic_tags": ["<tag1>", "<tag2>", "<tag3>"], '
                    '"density_map_description": "<where in the image this object likely is>"}'
                ),
                max_tokens=300,
            )
            tags = tag_data.get("semantic_tags", [tp.description])

            # Generate a simple density map (uniform for text prompts)
            density_map = [0.5] * (MASK_GRID_SIZE * MASK_GRID_SIZE)

            dense_features = DensePromptFeatures(
                source_type="text",
                embedding=text_emb,
                semantic_tags=tags,
                density_map=density_map,
            )
            logger.debug(f"📝 Text: '{tp.description}' | tags={tags}")

        # Dense: Mask 
        if sam_input.mask_prompt:
            mp = sam_input.mask_prompt
            flat_mask = [cell for row in mp.mask_grid for cell in row]
            # Embed mask as a 256-d vector (the flattened 16×16 grid)
            mask_emb = flat_mask[:EMBEDDING_DIMENSIONS] + [0.0] * max(0, EMBEDDING_DIMENSIONS - len(flat_mask))
            dense_features = DensePromptFeatures(
                source_type="mask",
                embedding=mask_emb,
                semantic_tags=["coarse_mask_hint"],
                density_map=flat_mask,
            )
            logger.debug("🎭 Mask hint encoded")

        # Unified prompt embedding 
        if sam_input.prompt_type == PromptType.EVERYTHING:
            # "Segment everything" — no prompt, use zero embedding
            prompt_emb = [0.0] * EMBEDDING_DIMENSIONS
            prompt_summary = "Segment all objects in the image (no specific prompt)."
        elif dense_features:
            prompt_emb = dense_features.embedding[:EMBEDDING_DIMENSIONS]
            prompt_summary = (
                f"{'Text' if dense_features.source_type == 'text' else 'Mask'} prompt: "
                f"{sam_input.text_prompt.description if sam_input.text_prompt else 'coarse mask hint'}"
            )
        elif sparse_tokens:
            # Mean-pool sparse token embeddings → pad/truncate to 1536-d
            sp_matrix = np.array([t.embedding for t in sparse_tokens], dtype=np.float64)
            sp_mean = np.mean(sp_matrix, axis=0)
            # Repeat to fill 1536-d
            repeats = math.ceil(EMBEDDING_DIMENSIONS / len(sp_mean))
            prompt_emb = np.tile(sp_mean, repeats)[:EMBEDDING_DIMENSIONS].tolist()
            prompt_summary = (
                f"{len(sparse_tokens)} sparse prompt token(s): "
                f"{[t.token_type for t in sparse_tokens]}"
            )
        else:
            prompt_emb = [0.0] * EMBEDDING_DIMENSIONS
            prompt_summary = "No prompt specified."

        elapsed = time.perf_counter() - t0
        result  = PromptEncoderResult(
            prompt_type=sam_input.prompt_type,
            sparse_tokens=sparse_tokens,
            dense_features=dense_features,
            prompt_summary=prompt_summary,
            prompt_embedding=prompt_emb,
            processing_time=round(elapsed, 4),
        )
        logger.info(
            f"✅ [PROMPT ENCODER] sparse_tokens={len(sparse_tokens)} | "
            f"dense={'yes' if dense_features else 'no'} | summary='{prompt_summary[:60]}' | time={elapsed:.4f}s"
        )
        return result

class ImageEncoderStage:
    """
    Stage: IMAGE ENCODER
    Extracts dense patch-level features from the image using a ViT-style patch decomposition.

    Architecture fidelity:
    - SAM's image encoder is a ViT-H (huge) with:
      * 14×14 px patch size on 1024×1024 input → 64×64 patch grid
      * 1280-d patch embeddings
      * 32 transformer blocks with window attention
    - We simulate this with a 16×16 patch grid and 256-d patch embeddings extracted via GPT-4.1 vision structured analysis.
    - Each patch gets: description, salience score, object label, feature vector.
    - The image encoder runs ONCE and its output is reused by both Image Embedding and Feature Correlation stages.
    """

    def run(self, b64_image: str, mime_type: str, agent: BaseAIAgent) -> ImageEncoderResult:
        logger.info(f"⚙️ [IMAGE ENCODER] Extracting ViT-style patch features ({PATCH_GRID_H}×{PATCH_GRID_W} grid)...")
        t0 = time.perf_counter()

        #  Step 1: High-level scene analysis 
        scene_data = agent._gpt_json_response(
            system=(
                "You are a ViT image encoder for the Segment Anything Model. "
                "Analyse the image to identify all objects, their spatial distribution, and dominant visual features. JSON only."
            ),
            user=[
                {
                    "type"     : "image_url",
                    "image_url": {
                        "url"   : f"data:{mime_type};base64,{b64_image}",
                        "detail": VISION_DETAIL,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        "Analyse this image for segmentation.\n\n"
                        "Respond with:\n"
                        "{\n"
                        '  "scene_summary": "<2-sentence summary>",\n'
                        '  "dominant_objects": ["<obj1>", "<obj2>", "<obj3>"],\n'
                        '  "spatial_regions": [\n'
                        "    {\n"
                        '      "region": "<top-left|top-right|center|bottom-left|bottom-right|full>",\n'
                        '      "content": "<what is in this region>",\n'
                        '      "salience": <0.0-1.0>,\n'
                        '      "object_label": "<primary object>"\n'
                        "    }\n"
                        "  ]\n"
                        "}"
                    ),
                },
            ],
            max_tokens =1500,
            temperature=0.1,
        )

        scene_summary = scene_data.get("scene_summary", "")
        dominant_objects = scene_data.get("dominant_objects", [])
        spatial_regions = scene_data.get("spatial_regions", [])

        # Step 2: Simulate 16×16 patch grid 
        # Map spatial regions to patch grid positions; fill remaining patches with background features.
        rng = np.random.default_rng(seed=int(hashlib.md5(b64_image[:32].encode()).hexdigest(), 16) % (2**32))

        # Build region → grid mapping
        region_map = {
            "top-left": (0,  0,  8,  8),
            "top-right": (0,  8,  8, 16),
            "center": (4,  4, 12, 12),
            "bottom-left": (8,  0, 16,  8),
            "bottom-right": (8,  8, 16, 16),
            "full": (0,  0, 16, 16),
        }

        # Assign salience and labels to patch grid cells
        patch_salience = np.zeros((PATCH_GRID_H, PATCH_GRID_W), dtype=np.float32)
        patch_labels = [["background"] * PATCH_GRID_W for _ in range(PATCH_GRID_H)]
        patch_descs = [["background region"] * PATCH_GRID_W for _ in range(PATCH_GRID_H)]

        for sr in spatial_regions:
            bounds = region_map.get(sr.get("region", "full"), (0, 0, 16, 16))
            r1, c1, r2, c2 = bounds
            for r in range(r1, r2):
                for c in range(c1, c2):
                    patch_salience[r][c]  = float(sr.get("salience", 0.5))
                    patch_labels[r][c]    = sr.get("object_label", "object")
                    patch_descs[r][c]     = sr.get("content", "")

        # Step 3: Generate patch embeddings 
        patches: List[ImagePatch] = []
        for row in range(PATCH_GRID_H):
            for col in range(PATCH_GRID_W):
                patch_id = f"P{row:02d}{col:02d}"
                x_norm   = (col + 0.5) / PATCH_GRID_W
                y_norm   = (row + 0.5) / PATCH_GRID_H
                sal      = float(patch_salience[row][col])

                # Patch feature: position encoding + salience-weighted noise
                pos_enc = positional_embedding_2d(x_norm, y_norm, dim=IMAGE_ENCODER_EMBED_DIM)
                noise = rng.normal(0, 0.01 * (1 - sal + 0.1), IMAGE_ENCODER_EMBED_DIM)
                feature = (np.array(pos_enc, dtype=np.float64) + noise)
                # Normalise
                norm_f = np.linalg.norm(feature)
                if norm_f > 0:
                    feature /= norm_f

                patches.append(ImagePatch(
                    patch_id=patch_id,
                    row=row,
                    col=col,
                    x_norm=x_norm,
                    y_norm=y_norm,
                    description=patch_descs[row][col],
                    feature=feature.tolist(),
                    salience=round(sal, 4),
                    object_label=patch_labels[row][col],
                ))

        elapsed = time.perf_counter() - t0
        result = ImageEncoderResult(
            model=CHAT_MODEL,
            patches=patches,
            total_patches=len(patches),
            dominant_objects=dominant_objects,
            scene_summary=scene_summary,
            processing_time=round(elapsed, 4),
        )
        logger.info(
            f"✅ [IMAGE ENCODER] patches={len(patches)} ({PATCH_GRID_H}×{PATCH_GRID_W}) | "
            f"objects={dominant_objects[:4]} | time={elapsed:.4f}s"
        )
        return result

class ImageEmbeddingStage:
    """
    Stage: IMAGE EMBEDDING
    Fuses all patch features into a single dense image embedding and computes a prompt-guided spatial attention map.

    Architecture fidelity:
    - In SAM, the image encoder output is a (H/16 × W/16 × 256) feature tensor. The image embedding is what the mask decoder attends to.
    - We produce:
      (a) A global 1536-d image embedding (full scene API call)
      (b) A 256-d mean-pooled patch vector (across all patch features)
      (c) A 16×16 spatial attention map guided by the prompt embedding
    """

    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def run(self, prompt_encoder: PromptEncoderResult, image_encoder: ImageEncoderResult, agent: BaseAIAgent) -> ImageEmbeddingResult:
        logger.info("⚙️ [IMAGE EMBEDDING] Computing global + patch embeddings...")
        t0 = time.perf_counter()

        # Global image embedding 
        emb_resp = agent._retry_api_call(
            self._client.embeddings.create,
            model = EMBEDDING_MODEL,
            input = image_encoder.scene_summary + " " + " ".join(image_encoder.dominant_objects),
        )
        global_emb = emb_resp.data[0].embedding

        # Pooled patch vector 
        patch_matrix = np.array(
            [p.feature for p in image_encoder.patches], dtype=np.float64
        )
        pooled_patch = np.mean(patch_matrix, axis=0).tolist()

        #  Spatial attention map (prompt-guided) ─────────────────────────
        # Compute attention weight for each patch as:
        # attention = salience × prompt_patch_similarity
        prompt_emb_arr = np.array(
            prompt_encoder.prompt_embedding[:IMAGE_ENCODER_EMBED_DIM],
            dtype=np.float64,
        )
        prompt_norm = np.linalg.norm(prompt_emb_arr)

        spatial_attention: List[SpatialAttentionCell] = []
        attn_raw = []
        for patch in image_encoder.patches:
            feat  = np.array(patch.feature, dtype=np.float64)
            if prompt_norm > 0:
                sim = float(np.dot(prompt_emb_arr, feat) / (prompt_norm * (np.linalg.norm(feat) + 1e-10)))
            else:
                sim = patch.salience
            # Attention = 0.6×salience + 0.4×similarity (weighted blend)
            attn = 0.6 * patch.salience + 0.4 * max(0.0, sim)
            attn_raw.append(attn)

        # Normalise to [0, 1]
        attn_arr = np.array(attn_raw)
        if attn_arr.max() > 0:
            attn_arr = attn_arr / attn_arr.max()

        for i, patch in enumerate(image_encoder.patches):
            spatial_attention.append(SpatialAttentionCell(
                row=patch.row, col=patch.col, attention=round(float(attn_arr[i]), 4)
            ))

        # High-salience patches (top 20%)
        threshold = float(np.percentile(attn_arr, 80))
        high_sal = [
            image_encoder.patches[i].patch_id
            for i, v in enumerate(attn_arr)
            if v >= threshold
        ]

        elapsed = time.perf_counter() - t0
        result  = ImageEmbeddingResult(
            global_embedding=global_emb,
            patch_embedding_dim=IMAGE_ENCODER_EMBED_DIM,
            pooled_patch_vec=pooled_patch,
            spatial_attention=spatial_attention,
            high_salience_patches=high_sal,
            processing_time=round(elapsed, 4),
        )
        logger.info(
            f"✅ [IMAGE EMBEDDING] global_dim={len(global_emb)} | pooled_dim={len(pooled_patch)} | "
            f"high_salience={len(high_sal)}/{len(image_encoder.patches)} patches | time={elapsed:.4f}s"
        )
        return result

class MaskDecoderStage:
    """
    Stage: MASK DECODER
    Decodes the fused prompt + image embeddings into binary segmentation masks.

    Architecture fidelity:
    - SAM's mask decoder is a 2-layer transformer decoder that:
      * Cross-attends prompt tokens to image embedding
      * Produces 3 mask candidates at low resolution (4× downsampled)
      * Predicts IoU scores for each candidate
      * Upsamples to original resolution via transposed convolutions
    - We produce:
      * 3 mask candidates (NUM_MASK_CANDIDATES) ranked by IoU estimate
      * Each mask: 16×16 binary grid + RLE encoding + IoU + stability score
      * Masks are described spatially so GPT-4.1 can generate realistic grids
    """

    def run(self, prompt_encoder: PromptEncoderResult, image_encoder: ImageEncoderResult, image_embedding: ImageEmbeddingResult, b64_image: str, mime_type: str, agent: BaseAIAgent) -> MaskDecoderResult:
        logger.info(f"⚙️  [MASK DECODER] Decoding {NUM_MASK_CANDIDATES} mask candidates...")
        t0 = time.perf_counter()

        # High-salience context for decoder
        high_sal_patches = [
            p for p in image_encoder.patches
            if p.patch_id in image_embedding.high_salience_patches
        ]
        region_desc = "; ".join(
            f"{p.object_label} at ({p.x_norm:.2f},{p.y_norm:.2f})"
            for p in high_sal_patches[:8]
        )

        # GPT-4.1 mask decoding 
        decode_data = agent._gpt_json_response(
            system=(
                "You are the mask decoder of the Segment Anything Model (SAM). "
                "Given prompt and image context, predict 3 segmentation mask candidates with different confidence levels. "
                "Describe the spatial extent of each mask accurately. JSON only."
            ),
            user=[
                {
                    "type"     : "image_url",
                    "image_url": {
                        "url"   : f"data:{mime_type};base64,{b64_image}",
                        "detail": VISION_DETAIL,
                    },
                },
                {
                    "type": "text",
                    "text": (
                        f"Prompt: {prompt_encoder.prompt_summary}\n"
                        f"Scene: {image_encoder.scene_summary}\n"
                        f"Objects: {', '.join(image_encoder.dominant_objects)}\n"
                        f"High-attention regions: {region_desc}\n\n"
                        f"Generate {NUM_MASK_CANDIDATES} segmentation mask candidates "
                        f"(best to worst quality).\n\n"
                        "Respond with:\n"
                        "{\n"
                        '  "masks": [\n'
                        "    {\n"
                        '      "rank": <1|2|3>,\n'
                        '      "iou_estimate": <0.0-1.0>,\n'
                        '      "stability_score": <0.0-1.0>,\n'
                        '      "confidence": <0.0-1.0>,\n'
                        '      "object_class": "<primary object being segmented>",\n'
                        '      "object_description": "<detailed description>",\n'
                        '      "area_ratio": <0.0-1.0, fraction of image covered>,\n'
                        '      "regions": [\n'
                        "        {\n"
                        '          "region_id": "R001",\n'
                        '          "description": "<what is here>",\n'
                        '          "spatial_bounds": "<top-left quadrant etc>",\n'
                        '          "coverage_ratio": <0.0-1.0>,\n'
                        '          "x1": <0.0-1.0>, "y1": <0.0-1.0>,\n'
                        '          "x2": <0.0-1.0>, "y2": <0.0-1.0>\n'
                        "        }\n"
                        "      ]\n"
                        "    }\n"
                        "  ]\n"
                        "}"
                    ),
                },
            ],
            max_tokens =2500,
            temperature=0.1,
        )

        raw_masks = decode_data.get("masks", [])
        masks: List[SegmentationMask] = []

        for i, rm in enumerate(raw_masks[:NUM_MASK_CANDIDATES]):
            # Parse regions and generate binary mask grid
            raw_regions = rm.get("regions", [])
            mask_grid   = generate_mask_grid(raw_regions)
            rle = rle_encode(mask_grid)
            iou = float(rm.get("iou_estimate", max(0.0, 0.85 - i * 0.1)))
            stab = float(rm.get("stability_score", max(0.0, 0.90 - i * 0.08)))
            conf = float(rm.get("confidence", max(0.0, 0.88 - i * 0.1)))
            area = float(rm.get("area_ratio", 0.2))

            regions = [
                MaskRegion(
                    region_id= r.get("region_id", f"R{j:03d}"),
                    description= r.get("description", ""),
                    spatial_bounds= r.get("spatial_bounds", ""),
                    coverage_ratio= float(r.get("coverage_ratio", 0.5)),
                )
                for j, r in enumerate(raw_regions)
            ]

            mask = SegmentationMask(
                mask_id=f"MASK_{i+1:03d}",
                rank=rm.get("rank", i + 1),
                iou_estimate=round(iou, 4),
                stability_score=round(stab, 4),
                confidence=round(conf, 4),
                rle_encoding=rle,
                mask_grid=mask_grid,    
                regions=regions,
                area_ratio=round(area, 4),
                object_class=rm.get("object_class", "object"),
                object_description=rm.get("object_description", ""),
                is_valid=(iou >= IOU_THRESHOLD and stab >= STABILITY_THRESHOLD),
            )
            masks.append(mask)
            icon = "✅" if mask.is_valid else "⚠️ "
            logger.debug(
                f"{icon} Mask[{i+1}] {mask.object_class} | "
                f"iou={iou:.4f} stab={stab:.4f} area={area:.4f} | valid={mask.is_valid}"
            )

        #  Ensure at least one mask exists ───────────────────────────────
        if not masks:
            logger.warning("⚠️  [MASK DECODER] No masks returned — generating fallback")
            fallback_grid = generate_mask_grid([{"x1": 0.25, "y1": 0.25, "x2": 0.75, "y2": 0.75}])
            masks.append(SegmentationMask(
                mask_id="MASK_001", rank=1,
                iou_estimate=0.5, stability_score=0.5, confidence=0.5,
                rle_encoding=rle_encode(fallback_grid),
                mask_grid=fallback_grid, regions=[],
                area_ratio=0.25,
                object_class="unknown",
                object_description="Fallback centre-crop mask",
                is_valid=False,
            ))

        best_mask = min(masks, key=lambda m: m.rank)
        valid_count = sum(1 for m in masks if m.is_valid)
        elapsed = time.perf_counter() - t0

        result = MaskDecoderResult(
            masks=masks,
            best_mask_id=best_mask.mask_id,
            num_valid_masks=valid_count,
            decoding_strategy="cross_attention_transformer_simulation",
            processing_time=round(elapsed, 4),
        )
        logger.info(
            f"✅ [MASK DECODER] masks={len(masks)} | valid={valid_count} | "
            f"best={best_mask.mask_id} iou={best_mask.iou_estimate:.4f} | time={elapsed:.4f}s"
        )
        logger.debug(f"🔍 Best mask regions: {[r.description for r in best_mask.regions[:3]]}")
        return result

class FeatureCorrelationStage:
    """
    Stage: FEATURE CORRELATION
    Computes a cross-correlation matrix between prompt token embeddings and image patch features.

    Architecture fidelity:
    - In SAM's mask decoder, cross-attention between prompt tokens (queries) and image patches (keys/values) determines which regions the mask should cover.
    - We compute explicit cosine similarities between:
      * Each sparse prompt token embedding (64-d, tiled to 256-d)
      * Each image patch feature vector (256-d)
    - The top-k most correlated patches indicate WHERE the model should place the mask — directly interpretable attention.
    """

    def run(self, prompt_encoder: PromptEncoderResult, image_encoder: ImageEncoderResult, image_embedding: ImageEmbeddingResult) -> FeatureCorrelationResult:
        logger.info(f"⚙️ [FEATURE CORRELATION] Computing prompt × patch correlations...")
        t0 = time.perf_counter()

        # Build prompt token list (sparse tokens + dense as single token)
        prompt_tokens = []
        for st in prompt_encoder.sparse_tokens:
            # Tile 64-d sparse embedding → 256-d to match patch features
            emb_arr = np.array(st.embedding, dtype=np.float64)
            repeats = math.ceil(IMAGE_ENCODER_EMBED_DIM / len(emb_arr))
            tiled = np.tile(emb_arr, repeats)[:IMAGE_ENCODER_EMBED_DIM]
            prompt_tokens.append({"id": st.token_id, "embedding": tiled})

        if prompt_encoder.dense_features:
            dense_emb = np.array(
                prompt_encoder.dense_features.embedding[:IMAGE_ENCODER_EMBED_DIM],
                dtype=np.float64,
            )
            prompt_tokens.append({"id": "DENSE_PROMPT", "embedding": dense_emb})

        if not prompt_tokens:
            # "Segment everything" — use global image embedding as prompt
            glob_arr = np.array(
                prompt_encoder.prompt_embedding[:IMAGE_ENCODER_EMBED_DIM],
                dtype=np.float64,
            )
            prompt_tokens.append({"id": "GLOBAL", "embedding": glob_arr})

        #  Compute correlation matrix ────────────────────────────────────
        correlations: List[PatchCorrelation] = []
        for pt in prompt_tokens:
            pt_arr = pt["embedding"]
            pt_norm = np.linalg.norm(pt_arr)
            for patch in image_encoder.patches:
                p_arr  = np.array(patch.feature, dtype=np.float64)
                p_norm = np.linalg.norm(p_arr)
                if pt_norm > 0 and p_norm > 0:
                    corr = float(np.dot(pt_arr, p_arr) / (pt_norm * p_norm))
                else:
                    corr = 0.0
                correlations.append(PatchCorrelation(
                    prompt_token_id = pt["id"],
                    patch_id        = patch.patch_id,
                    correlation     = round(corr, 4),
                    row             = patch.row,
                    col             = patch.col,
                ))

        #  Top-k most correlated patches (across all prompt tokens) ───────
        sorted_corrs = sorted(correlations, key=lambda c: c.correlation, reverse=True)
        seen_patches: Set[str] = set()
        top_k_patches: List[str] = []
        for c in sorted_corrs:
            if c.patch_id not in seen_patches:
                top_k_patches.append(c.patch_id)
                seen_patches.add(c.patch_id)
            if len(top_k_patches) >= CORRELATION_TOP_K:
                break

        all_corr_vals = [c.correlation for c in correlations]
        mean_corr = float(np.mean(all_corr_vals)) if all_corr_vals else 0.0
        max_corr = float(np.max(all_corr_vals)) if all_corr_vals else 0.0

        # Prompt-patch alignment: fraction of top-k in high-salience patches
        high_sal_set = set(image_embedding.high_salience_patches)
        overlap = len([p for p in top_k_patches if p in high_sal_set])
        alignment = overlap / max(len(top_k_patches), 1)

        elapsed = time.perf_counter() - t0
        result  = FeatureCorrelationResult(
            correlations=correlations,
            top_k_patches=top_k_patches,
            mean_correlation=round(mean_corr, 4),
            max_correlation=round(max_corr, 4),
            prompt_patch_alignment=round(alignment, 4),
            processing_time=round(elapsed, 4),
        )
        logger.info(
            f"✅ [FEATURE CORRELATION] correlations={len(correlations)} | top_k={top_k_patches[:4]} | "
            f"mean_corr={mean_corr:.4f} | max_corr={max_corr:.4f} | alignment={alignment:.4f} | time={elapsed:.4f}s"
        )
        return result
    
# ==========================================
# MLM AGENT  —  Orchestrates all 8 pipeline stages
# ==========================================
class SAMAgent(BaseAIAgent):
    """
    Masked Language Model Agent

    Pipeline:
    [Text Input] → [Token Masking] → [Embedding Layer] →
    [Left Context] ⟷ [Right Context] → [Bidirectional Attention] →
    [Masked Token Prediction] → [Feature Representation]

    Core Principle: True bidirectionality — every masked position attends to both left and right context. Non-autoregressive (BERT-style).
    """
    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self._loader = ImageLoader()
        self._prompt_encoder = PromptEncoderStage(client)
        self._image_encoder = ImageEncoderStage()
        self._image_embedding = ImageEmbeddingStage(client)
        self._mask_decoder = MaskDecoderStage()
        self._feature_corr = FeatureCorrelationStage()

    # Public entry point 

    def process(self, sam_input: SAMInput) -> SAMOutput:
        """
        Execute the full SAM pipeline.

        Args:
            sam_input: Validated SAMInput with image + prompt.

        Returns:
            SAMOutput: Fully structured segmentation result with RLE masks.

        Raises:
            ValueError: On invalid input or image.
            RuntimeError: On unrecoverable API failure.
        """
        pipeline_start = time.perf_counter()
        logger.info(
            f"🚀 [SAM AGENT] Pipeline START | request_id={sam_input.request_id}"
            f"📥 [INPUT] image={sam_input.image.source_type.value} | "
            f"prompt={sam_input.prompt_type.value} | multimask={sam_input.multimask_output}"     
        )

        try:
            # Load & validate image 
            b64_image, mime_type, sha256 = self._loader.load(sam_input.image)
            logger.info(
                "⚙️ [IMAGE INPUT] Loading and fingerprinting image..."
                f"✅ [IMAGE INPUT] mime={mime_type} | "
                f"b64_len={len(b64_image)} | sha256={sha256[:12]}..."
            )

            # Stage: Prompt Encoder 
            prompt_enc = checkpointer.load("PROMPT_ENCODER", PromptEncoderResult)
            if not prompt_enc:
                prompt_enc = self._prompt_encoder.run(sam_input, self)
                checkpointer.save("PROMPT_ENCODER", prompt_enc)

            # Stage: Image Encoder 
            image_enc = checkpointer.load("IMAGE_ENCODER", ImageEncoderResult)
            if not image_enc:
                image_enc = self._image_encoder.run(b64_image, mime_type, self)
                checkpointer.save("IMAGE_ENCODER", image_enc)

            # Stage: Image Embedding 
            image_emb = checkpointer.load("IMAGE_EMBEDDING", ImageEmbeddingResult)
            if not image_emb:
                image_emb = self._image_embedding.run(prompt_enc, image_enc, self)
                checkpointer.save("IMAGE_EMBEDDING", image_emb)

            # Stage: Mask Decoder ‖ Feature Correlation (parallel) 
            mask_result = checkpointer.load("MASK_DECODER", MaskDecoderResult)
            feat_corr_result = checkpointer.load("FEATURE_CORRELATION", FeatureCorrelationResult)
            
            if not mask_result or not feat_corr_result:
                logger.info("⚙️ [SAM AGENT] Running Mask Decoder ‖ Feature Correlation (parallel)...")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    future_mask = None
                    future_corr = None
                    
                    if not mask_result:
                        future_mask = executor.submit(
                            self._mask_decoder.run,
                            prompt_enc, image_enc, image_emb, b64_image, mime_type, self
                        )
                    if not feat_corr_result:
                        future_corr = executor.submit(
                            self._feature_corr.run,
                            prompt_enc, image_enc, image_emb
                        )
                    
                    futures = [f for f in (future_mask, future_corr) if f is not None]
                    for future in as_completed(futures):
                        if future_mask and future is future_mask:
                            mask_result = future.result()
                            checkpointer.save("MASK_DECODER", mask_result)
                        elif future_corr and future is future_corr:
                            feat_corr_result = future.result()
                            checkpointer.save("FEATURE_CORRELATION", feat_corr_result)
            else:
                logger.info("✅ [SAM AGENT] Loaded Mask Decoder & Feature Correlation from checkpoint.")

            # Stage: Segmentation Output 
            best_mask  = next(
                (m for m in mask_result.masks if m.mask_id == mask_result.best_mask_id),
                mask_result.masks[0]
            )

            status = (
                ProcessingStatus.SUCCESS if mask_result.num_valid_masks > 0
                else ProcessingStatus.PARTIAL
            )

            seg_summary = self._gpt_text_response(
                system=(
                    "You are a segmentation result reporter for SAM. "
                    "Write a precise, concise segmentation summary."
                ),
                user=(
                    f"Prompt: {prompt_enc.prompt_summary}\n"
                    f"Best mask: {best_mask.object_class} | "
                    f"IoU={best_mask.iou_estimate:.4f} | "
                    f"Stability={best_mask.stability_score:.4f} | "
                    f"Area={best_mask.area_ratio:.4f} of image\n"
                    f"Description: {best_mask.object_description}\n"
                    f"Valid masks: {mask_result.num_valid_masks}/{len(mask_result.masks)}\n"
                    f"Prompt-patch alignment: {feat_corr_result.prompt_patch_alignment:.4f}\n\n"
                    "Write a 2-sentence segmentation result summary."
                ),
                max_tokens=200,
            )

            total_time = time.perf_counter() - pipeline_start
            output = SAMOutput(
                request_id=sam_input.request_id,
                status=status,
                prompt_encoder=prompt_enc,
                image_encoder=image_enc,
                image_embedding=image_emb,
                mask_decoder=mask_result,
                feature_correlation=feat_corr_result,
                best_mask=best_mask,
                segmentation_summary=seg_summary,
                total_pipeline_time=round(total_time, 4),
                metadata={
                    **sam_input.metadata,
                    "model": CHAT_MODEL,
                    "image_sha256": sha256,
                    "prompt_type": sam_input.prompt_type.value,
                },
            )

            logger.info(
                f"🎉 [SAM AGENT] Pipeline COMPLETE | total_time={total_time:.4f}s | "
                f"status={status.value} | best_iou={best_mask.iou_estimate:.4f} | "
                f"valid_masks={mask_result.num_valid_masks}/{len(mask_result.masks)} | alignment={feat_corr_result.prompt_patch_alignment:.4f}"
            )
            return output

        except Exception as e:
            elapsed = time.perf_counter() - pipeline_start
            logger.error(
                f"❌ [SAM AGENT] Pipeline FAILED after {elapsed:.4f}s | "
                f"error={type(e).__name__}: {e}"
            )
            raise RuntimeError(f"❌ SAM pipeline failed: {e}") from e

    # Convenience factory methods 

    @staticmethod
    def text_prompt_input(image_source: str, description: str, source_type: ImageSource = ImageSource.URL, mime_type: str = "image/jpeg", **kwargs) -> SAMInput:
        return SAMInput(
            image=ImagePayload(
                source_type=source_type, source=image_source, mime_type=mime_type
            ),
            prompt_type=PromptType.TEXT,
            text_prompt=TextPrompt(description=description),
            **kwargs,
        )

    @staticmethod
    def point_prompt_input(image_source: str, points: List[Tuple[float, float, int]], source_type: ImageSource = ImageSource.URL, mime_type: str = "image/jpeg", **kwargs) -> SAMInput:
        return SAMInput(
            image=ImagePayload(
                source_type=source_type, source=image_source, mime_type=mime_type
            ),
            prompt_type=PromptType.POINT,
            point_prompts=[PointPrompt(x=x, y=y, label=lbl) for x, y, lbl in points],
            **kwargs,
        )

    @staticmethod
    def box_prompt_input(image_source: str, x1: float, y1: float, x2: float, y2: float, source_type: ImageSource = ImageSource.URL, mime_type: str = "image/jpeg", **kwargs) -> SAMInput:
        return SAMInput(
            image=ImagePayload(
                source_type=source_type, source=image_source, mime_type=mime_type
            ),
            prompt_type=PromptType.BOX,
            box_prompt=BoxPrompt(x1=x1, y1=y1, x2=x2, y2=y2),
            **kwargs,
        )

    # Display helper 

    def display_output(self, output: SAMOutput) -> None:
        div = "=" * 80
        print(f"\n{div}")
        print("🟢 SAM AGENT — Segment Anything Model Pipeline Result")
        print(f"{div}")
        print(f"Request ID: {output.request_id}")
        print(f"Status: {output.status.value}")
        print(f"Total Time: {output.total_pipeline_time}s")
        
        print(f"\n{div}")
        print(f"  🔗 FEATURE CORRELATION (parallel with Mask Decoder)")
        fc = output.feature_correlation
        print(f"Total Corr Pairs: {len(fc.correlations)}")
        print(f"Mean Correlation: {fc.mean_correlation:.4f}")
        print(f"Max Correlation : {fc.max_correlation:.4f}")
        print(f"Top-{CORRELATION_TOP_K} Patches : {fc.top_k_patches}")
        print(f"Prompt Alignment: {fc.prompt_patch_alignment:.4f}")
        # Show top-5 correlations
        top5 = sorted(fc.correlations, key=lambda c: c.correlation, reverse=True)[:5]
        print(f"Highest correlations:")
        for c in top5:
            bar = "█" * int(max(0, c.correlation) * 15)
            print(f"{c.prompt_token_id} → {c.patch_id} [{bar:<15}] {c.correlation:.4f}")
        print(f"Time: {fc.processing_time}s")

        print(f"{div}")
        print(f"📤 SEGMENTATION OUTPUT")
        print(f"\nBest Mask: {output.best_mask.mask_id}")
        print(f"Object: {output.best_mask.object_class}")
        print(f"IoU Estimate: {output.best_mask.iou_estimate:.4f}")
        print(f"Stability Score: {output.best_mask.stability_score:.4f}")
        print(f"Area Coverage: {output.best_mask.area_ratio:.2%} of image")
        print(f"RLE Encoding: {output.best_mask.rle_encoding[:60]}...")
        print(f"\nSummary:\n  {output.segmentation_summary}")
        print(f"\n{div}\n")

# ==========================================
# Instatiation
# ==========================================
def create_sam_agent(api_key: str = TOKEN, endpoint: str = ENDPOINT) -> SAMAgent:
    """Factory function to create an instance of SAMAgent with the provided API credentials."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    logger.info("🏭 [FACTORY] SAMAgent instantiated and ready.")
    return SAMAgent(client)

# ==========================================
# Example usage
# ==========================================
if __name__ == "__main__":
    agent = create_sam_agent()

    # ═══════════════════════════════════════════════════════════════════════
    # DEMO 1: Text Prompt — "segment everything dog-like"
    # ═══════════════════════════════════════════════════════════════════════
    # print("═" * 72)
    # print("Demo 1: TEXT PROMPT segmentation")
    # print("═" * 72)

    # sam_input = SAMAgent.text_prompt_input(
    #     image_source="assets/image/husky.png",
    #     description="the primary foreground object",
    #     source_type=ImageSource.FILE,
    #     mime_type="image/png",
    #     metadata={"source": "sam_agent_demo_1", "version": "1.0"},
    # )
    # result = agent.process(sam_input)
    # agent.display_output(result)

    # ═══════════════════════════════════════════════════════════════════════
    # DEMO 2: Point Prompt — foreground click at image centre
    # ═══════════════════════════════════════════════════════════════════════
    # print("═" * 72)
    # print("Demo 2: POINT PROMPT segmentation")
    # print("═" * 72)

    # sam_point = SAMAgent.point_prompt_input(
    #     image_source="assets/image/apple.png",
    #     points=[
    #         (0.5, 0.5, PointLabel.FOREGROUND),   # click on centre
    #         (0.05, 0.05, PointLabel.BACKGROUND),  # background hint
    #     ],
    #     source_type =ImageSource.FILE,
    #     mime_type="image/png",
    #     metadata={"source": "sam_agent_demo_point_2", "version": "1.0"},
    # )
    # result2 = agent.process(sam_point)
    # print(f"Best mask IoU: {result2.best_mask.iou_estimate:.4f}")
    # print(f"Object Class: {result2.best_mask.object_class}")
    # print(f"Summary: {result2.segmentation_summary}\n")

    # ═══════════════════════════════════════════════════════════════════════
    # DEMO 3: Box Prompt — bounding box around top-half of image
    # ═══════════════════════════════════════════════════════════════════════
    print("═" * 72)
    print("Demo 3: BOX PROMPT segmentation")
    print("═" * 72)

    sam_box = SAMAgent.box_prompt_input(
        image_source="assets/image/airplane.png",
        x1=0.1, y1=0.1, x2=0.9, y2=0.9,
        source_type=ImageSource.FILE,
        mime_type="image/png",
        metadata={"source": "sam_agent_demo_box_3", "version": "1.0"},
    )
    result3 = agent.process(sam_box)
    print(f"Best mask IoU: {result3.best_mask.iou_estimate:.4f}")
    print(f"Stability Score: {result3.best_mask.stability_score:.4f}")
    print(f"Area Ratio: {result3.best_mask.area_ratio:.2%}")
    print(f"Valid Masks: {result3.mask_decoder.num_valid_masks}/{len(result3.mask_decoder.masks)}")
    print(f"Alignment: {result3.feature_correlation.prompt_patch_alignment:.4f}")
    print(f"Summary: {result3.segmentation_summary}\n")