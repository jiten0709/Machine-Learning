"""
Lightweight LLM pipeline using GitHub-hosted models.

Pipeline: 
LLMInput → TokenizationStage → EmbeddingStage → TransformerStage → LLMOutput.

Purpose
- Validate input (Pydantic), tokenize (tiktoken), produce embeddings, run transformer completions, and return a structured LLMOutput.

Highlights
- Uses GitHub models configured via env vars: GITHUB_MODEL_NAME, GITHUB_EMBED_NAME, authenticated with GITHUB_TOKEN at GITHUB_ENDPOINT.
- Resilience: exponential-backoff retry for API calls (rate limits, timeouts, API errors).
- Observability: structured logging, per-stage timings, and metadata.
- Usage: run as a script (python3 LLM.py) or instantiate via create_llm_agent(api_key, endpoint).
"""

import os
from openai import OpenAI, APIError, RateLimitError, APITimeoutError
import time
from abc import ABC, abstractmethod
from typing import Any, List, Dict
from pydantic import BaseModel, Field, field_validator, model_validator
import uuid
from datetime import datetime, timezone
import tiktoken

from dotenv import load_dotenv
load_dotenv()

from logging_setup import get_logger
logger = get_logger(__name__, log_file="llm.log")

# ==========================================
# Variable Configuration
# ==========================================
TOKEN = os.environ['GITHUB_TOKEN']
ENDPOINT = os.environ['GITHUB_ENDPOINT']
CHAT_MODEL = os.environ['GITHUB_MODEL_NAME']
EMBEDDING_MODEL = os.environ['GITHUB_EMBED_NAME']
ENCODING = "cl100k_base" # encoding used by gpt-4o / gpt-4 family
MAX_RETRIES = 3
MAX_TOKEN_LIMIT = 8192
EMBEDDING_DIMENSIONS = 1536
RETRY_BACKOFF_BASE = 2.0

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
        self.logger = get_logger(__name__, log_file="llm.log")
    
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
# Data Contracts (Pydantic models)
# ==========================================
class LLMInput(BaseModel):
    """Stage 1 — Validated raw input to the LLM pipeline."""

    request_id: str = Field(..., description="Unique identifier for the request.", default_factory=lambda: str(uuid.uuid4()))
    prompt: str = Field(..., description="The raw input text string from the user or system.")
    system_prompt: str = Field(
        default="You are a helpful, precise, and concise AI assistant.", 
        description="Optional system-level instructions to guide the model's behavior."
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Creativity threshold.")
    max_completion_tokens: int = Field(default=2048, ge=1, le=4096, description="Maximum tokens to generate in the output.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual information for processing.")

    @field_validator('prompt')
    @classmethod
    def strip_and_validate_prompt(cls, v: str) -> str:
        """Ensure prompt is not empty and strip extraneous whitespace."""
    
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Prompt cannot be empty or whitespace.")
        return cleaned
    
    @model_validator(mode='after')
    def attach_timestamp(self) -> 'LLMInput':
        """Attach a processing timestamp to the input model."""
        self.metadata['created_at'] = datetime.now(timezone.utc).isoformat()
        return self
 
class TokenizedResult(BaseModel):
    """Stage 2 — Tokenized representation of the input."""
    
    stage: str = 'TOKENIZATION'
    token_count: int = Field(..., description="Number of tokens generated from the input prompt.")
    tokens: List[int] = Field(..., description="List of token ids produced by the tokenizer.")
    truncated: bool = Field(default=False, description="Indicates if the input was truncated due to token limits.")
    processing_time: float = Field(..., description="Time taken to tokenize the input in seconds.")

class EmbeddingResult(BaseModel):
    """Stage 3 — Embedded vector representation of the tokenized input."""
    
    stage: str = 'EMBEDDING'
    model: str = Field(..., description="The embedding model used for vectorization.")
    dimensions: int = Field(..., description="Dimensionality of the embedding vectors.")
    embedding: List[float] = Field(..., description="The resulting embedding vector for the input.")
    processing_time: float = Field(..., description="Time taken to generate the embedding in seconds.")

    @field_validator('embedding')
    @classmethod
    def validate_embedding_length(cls, v: List[float]) -> List[float]:
        """Ensure the embedding vector has a reasonable length."""
        
        if not v:
            raise ValueError("Embedding vector cannot be empty.")
        return v

class TransformerResult(BaseModel):
    """Stage 4 — Raw output from the transformer model before final processing."""
    
    stage: str = 'TRANSFORMER'
    model: str = Field(..., description="The transformer model used for generation.")
    raw_response: str = Field(..., description="The raw text output generated by the transformer.")
    total_tokens: int = Field(..., description="Number of tokens generated in the output.")
    processing_time: float = Field(..., description="Time taken for the transformer to generate output in seconds.")

class LLMOutput(BaseModel):
    """Stage 5 — Structured output from the LLM pipeline."""
    request_id: str = Field(..., description="Unique identifier for the request, matching the input.")
    stage: str = 'OUTPUT'
    status: str = 'SUCCESS'
    llm_input: LLMInput
    tokenization: TokenizedResult
    embedding: EmbeddingResult
    transformer: TransformerResult
    response_text: str = Field(..., description="The final generated text.")
    total_pipeline_time: float = Field(..., description="Total time taken for the entire LLM processing pipeline in seconds.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual information about the output.")

    @model_validator(mode='after')
    def populate_output_metadata(self) -> 'LLMOutput':
        """Populate output metadata based on the input and processing stages."""
        
        self.metadata['completed_at'] = datetime.now(timezone.utc).isoformat()
        self.metadata['token_efficiency'] = round(
            len(self.response_text) / max(self.transformer.total_tokens, 1), 2
        )
        return self

# ==========================================
# Pipeline Stages as Data Contracts
# ==========================================
class TokenizationStage:
    """
    Stage 2: TOKENIZATION
    Converts raw text into token IDs using tiktoken (same tokenizer as GPT-4o).
    Applies truncation if the prompt exceeds MAX_TOKEN_LIMIT.
    """

    def __init__(self) -> None:
        self._encoding = tiktoken.get_encoding(ENCODING)

    def run(self, text: str) -> TokenizedResult:
        logger.info("⚙️ [TOKENIZATION] Starting tokenization stage...")
        t0 = time.perf_counter()

        tokens = self._encoding.encode(text)
        truncated = False
        if len(tokens) > MAX_TOKEN_LIMIT:
            logger.warning(
                f"🛡️ [TOKENIZATION] Token count {len(tokens)} exceeds "
                f"limit {MAX_TOKEN_LIMIT}. Truncating..."
            )
            tokens = tokens[:MAX_TOKEN_LIMIT]
            truncated = True

        elased_time = time.perf_counter() - t0
        result = TokenizedResult(
            token_count=len(tokens),
            tokens=tokens,
            truncated=truncated,
            processing_time=elased_time
        )
        logger.info(
            f"✅ [TOKENIZATION] Complete | tokens={result.token_count} | "
            f"truncated={truncated} | time={elased_time:.4f}s"
        )
        return result

class EmbeddingStage:
    """
    Stage 3: EMBEDDING
    Converts tokenized input into a high-dimensional vector using the specified embedding model.
    The vector captures meaning in latent space and is available downstream for similarity / retrieval tasks.
    """

    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def run(self, text: str, retry_fn) -> EmbeddingResult:
        logger.info("⚙️ [EMBEDDING] Starting semantic embedding vectorization...")
        t0 = time.perf_counter()

        response = retry_fn(
            self.client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=text
        )
        vector = response.data[0].embedding
        elased_time = time.perf_counter() - t0
        result = EmbeddingResult(
            model=EMBEDDING_MODEL,
            dimensions=EMBEDDING_DIMENSIONS,
            embedding=vector,
            processing_time=elased_time
        )
        logger.info(
            f"✅ [EMBEDDING] Complete | model={result.model} | dim={result.dimensions} | "
            f"time={elased_time:.4f}s"
        )
        return result
    
class TransformerStage:
    """
    Stage 4: TRANSFORMER
    The core attention-based generation engine.
    Executes the core LLM inference using the specified transformer model.
     - Accepts the original prompt and system instructions.
     - Returns the raw generated text and token usage for output structuring.
    """

    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def run(self, llm_input: LLMInput, retry_fn) -> TransformerResult:
        logger.info(
            f"⚙️  [TRANSFORMER] Invoking {CHAT_MODEL} | "
            f"temp={llm_input.temperature} | max_completion_tokens={llm_input.max_completion_tokens}"
        )
        t0 = time.perf_counter()
        
        response = retry_fn(
            self.client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": llm_input.system_prompt},
                {"role": "user", "content": llm_input.prompt}
            ],
            temperature=llm_input.temperature,
            max_completion_tokens=llm_input.max_completion_tokens
        )
        choice = response.choices[0]
        usage = response.usage.total_tokens
        elapsed_time = time.perf_counter() - t0
        result = TransformerResult(
            model=CHAT_MODEL,
            raw_response=choice.message.content or '',
            total_tokens=usage,
            processing_time=elapsed_time
        )
        logger.info(
            f"✅ [TRANSFORMER] Complete | model={result.model} | raw_response={result.raw_response[:50]}... | "
            f"total_tokens={result.total_tokens} | time={elapsed_time:.4f}s"
        )
        return result

# ==========================================
# Specialized Agent Implementation
# ==========================================
class LLMAgent(BaseAIAgent):
    """
    Standard Large Language Model Agent.
    Follows pipeline: [Input] -> [Tokenization] -> [Embedding] -> [Transformer] -> [Output]
    """
    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self._tokenizer = TokenizationStage()
        self._embedder = EmbeddingStage(client)
        self._transformer = TransformerStage(client)

    def process(self, llm_input: LLMInput) -> LLMOutput:
        """
        Execute the full LLM pipeline.

        Args:
            llm_input: Validated LLMInput pydantic model.

        Returns:
            LLMOutput: Fully structured pipeline result.

        Raises:
            ValueError: On invalid input.
            RuntimeError: On unrecoverable API failure.
        """
        pipeline_start = time.perf_counter()
        self.logger.info(
            f"🚀 [LLM AGENT] Pipeline START | "
            f"request_id={llm_input.request_id}"
        )
        try:
            # Stage 2: Tokenization
            tokenization = self._tokenizer.run(llm_input.prompt)

            # Stage 3: Embedding
            decoded_prompt = self._tokenizer._encoding.decode(tokenization.tokens)
            embedding = self._embedder.run(decoded_prompt, self._retry_api_call)

            # Stage 4: Transformer
            transformer = self._transformer.run(llm_input, self._retry_api_call)

            # Stage 5: Output Structuring
            total_time = time.perf_counter()-pipeline_start

            output = LLMOutput(
                request_id=llm_input.request_id,
                llm_input=llm_input,
                tokenization=tokenization,
                embedding=embedding,
                transformer=transformer,
                response_text=transformer.raw_response,
                total_pipeline_time=round(total_time, 4)
            )
            self.logger.info(
                f"🎉 [LLM AGENT] Pipeline COMPLETE | "
                f"request_id={output.request_id} | response_text={output.response_text[:50]}... | "
                f"total_tokens={transformer.total_tokens} | total_time={total_time:.4f}s "
            )
            return output
        except Exception as e:
            elapsed = time.perf_counter() - pipeline_start
            self.logger.error(
                f"💥 [LLM AGENT] Pipeline FAILED after {elapsed:.4f}s | "
                f"error={type(e).__name__}: {e}"
            )
            raise

    def display_output(self, output: LLMOutput) -> None:
        """Pretty-print a pipeline result to stdout."""
        divider = "=" * 100
        print(f"\n{divider}")
        print(f"  🤖 LLM AGENT — Pipeline Result (check logs for detailed information)")
        print(f"{divider}")

        print(f"👤 INPUT PROMPT")
        print(f"\nSystem prompt: {output.llm_input.system_prompt}\n")
        print(f"User prompt: {output.llm_input.prompt}\n")
        print(f"{divider}\n")

        print(f"🤖 LLM OUTPUT")
        print(f"\n{output.response_text}\n")
        print(f"{divider}\n")

# =========================================
# Instantiation
# =========================================
def create_llm_agent(api_key, endpoint) -> LLMAgent:
    """Factory function to create an instance of LLMAgent with the provided API credentials."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    logger.info("🏭 [FACTORY] LLMAgent instantiated and ready.")
    return LLMAgent(client)

# =========================================
# Entry Point
# ========================================
if __name__ == "__main__":
    # create agent instance
    agent = create_llm_agent(TOKEN, ENDPOINT)

    # build input
    llm_input = LLMInput(
        prompt="""Explain the self-attention mechanism in exactly 3 bullet points, each no longer than two sentences.""",
        system_prompt="""You are a senior machine learning researcher. Respond with precision and clarity. Use bullet points when instructed.""",
        temperature=0.4,
        max_completion_tokens=512,
        metadata={"source": "llm_agent_demo", "version": "1.0.0"},
    )

    # execute pipeline
    result = agent.process(llm_input)

    # display output
    agent.display_output(result)
