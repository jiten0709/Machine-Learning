"""LAM AGENT — Large Action Model

Pipeline: Input → Perception → Intent Recognition → Task Breakdown →
          [Action Planning ↔ Memory System ↔ Neuro-Symbolic Integration] →
          Feedback Integration → Output

Model: GPT-4.1

Core philosophy: 
LAM operates in ACTION SPACE, not token space. 
Every stage produces verifiable, executable, rule-validated artifacts.
LAM is designed to handle complex, multi-step tasks that require dynamic planning, memory management, and integration of symbolic reasoning. 
It serves as the "executive function" agent that can break down high-level goals into actionable steps, adapt its plan based on feedback, and maintain a memory of past actions and outcomes to inform future decisions.

Features:
- Production-grade | Pydantic v2 | ABC | State Checkpointing | Robust retry logic | Structured logging
"""

import time, uuid, json, os, textwrap, re
from abc import ABC, abstractmethod
from typing import Any, Deque, Callable, Dict, List, Optional, Tuple
from collections import deque
from openai import OpenAI, RateLimitError, APITimeoutError, APIError, APIConnectionError
from datetime import datetime, timezone
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    raise ImportError("❌ No env file found. Please create a .env file.")

from utils.logging_setup import get_logger
import logging
logger = get_logger(__name__, log_file="lam.log")
def _ltag(tag: str, level: int, msg: str, lgr: logging.Logger = logger) -> None:
    lgr.log(level, msg, extra={"tag": tag})

# ==========================================
# Variable Configuration
# ==========================================
from utils.state_checkpointer import StateCheckpointer
CHECKPOINT_DIR = Path("./checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
checkpointer = StateCheckpointer(
    directory=CHECKPOINT_DIR, 
    filename="lam_checkpoint.json",
    logger=logger
)

TOKEN = os.environ['GITHUB_TOKEN']
ENDPOINT = os.environ['GITHUB_ENDPOINT']
CHAT_MODEL = os.environ['GITHUB_MODEL_NAME']
EMBEDDING_MODEL = os.environ['GITHUB_EMBED_NAME']

MAX_RETRIES = 3
RETRY_BACKOFF_BASE = 2.0
MAX_FEEDBACK_ITERATIONS = 3 # max re-plan cycles in feedback loop
MEMORY_EPISODIC_LIMIT = 50 # max episodic memory slots
MEMORY_SEMANTIC_LIMIT = 100 # max semantic fact slots

PERCEPTION_PROMPT_TEMPLATE = """
Instruction: {instruction}
Environment: {environment}
Available Tools: {available_tools}
Constraints: {constraints}

Respond with:
{{
  "percepts": [
    {{
      "category": "<entity|action|resource|constraint|temporal>",
      "observation": "<what was observed>",
      "confidence": <0.0-1.0>,
      "source": "<instruction|environment|inferred>"
    }}
  ],
  "context_summary": "<2-sentence summary of what needs to happen>",
  "complexity_score": <0.0-1.0>
}}
"""

INTENT_PROMPT_TEMPLATE = """
Instruction: {instruction}
Context: {context_summary}
Percepts: {percepts_list}

Intent types available: {intent_types}

Respond with:
{{
  "intent_type": "<one of the intent types>",
  "primary_goal": "<clear one-sentence primary goal>",
  "sub_goals": [
    {{
      "goal_id": "SG001",
      "description": "<sub-goal>",
      "priority": <1-5>,
      "measurable": "<how to measure success>"
    }}
  ],
  "extracted_entities": ["<entity1>", "<entity2>"],
  "temporal_constraint": "<deadline or null>",
  "confidence": <0.0-1.0>
}}
"""

TASK_BREAKDOWN_PROMPT_TEMPLATE = """
Primary Goal: {primary_goal}
Sub-Goals:
{sub_goals_text}
Available Tools: {available_tools}
Max Steps: {max_steps}
Entities: {entities}

Produce a task graph. Each task must be atomic and independently executable.

Respond with:
{{
  "tasks": [
    {{
      "task_id": "T001",
      "name": "<short task name>",
      "description": "<what this task does>",
      "depends_on": ["<task_id>"],
      "required_tools": ["<tool>"],
      "estimated_cost": "<low|medium|high>",
      "is_critical": <true|false>
    }}
  ],
  "critical_path": ["T001", "T002", "..."]
}}
"""

ACTION_PLANNING_PROMPT_TEMPLATE = """
Goal: {primary_goal}
Tasks:
{tasks_text}
Available Tools: {available_tools}
Constraints: {constraints}
Memory Context: {memory_context}

For each task generate one or more actions.

Respond with:
{{
  "actions": [
    {{
      "task_id": "<parent task_id>",
      "step": <sequential integer>,
      "tool": "<tool_name>",
      "parameters": {{"key": "value"}},
      "expected_output": "<what this action produces>",
      "rationale": "<why this action is needed>"
    }}
  ],
  "estimated_total_steps": <integer>
}}
"""

ACTION_REPLAN_PROMPT_TEMPLATE = """
Goal: {primary_goal}
FAILED Actions (require correction):
{failed_actions}
Memory Context: {memory_context}
Constraints: {constraints}
Correction Hints: {corrections}

Generate replacement actions ONLY for the failed ones. Respond with:
{{
  "actions": [
    {{
      "task_id": "<task_id>",
      "step": <integer>,
      "tool": "<tool>",
      "parameters": {{"key": "value"}},
      "expected_output": "<expected>",
      "rationale": "<rationale>"
    }}
  ]
}}
"""

FEEDBACK_PROMPT_TEMPLATE = """
You are an action execution simulator. Simulate running the given action and determine if it would succeed. JSON only.

Goal: {primary_goal}
Action: {tool}({parameters})
Expected Output: {expected_output}
Rationale: {rationale}

Simulate this action and respond with:
{{
  "simulated_result": "<what the action produced or would produce>",
  "success": <true|false>,
  "correction": "<if failed, corrective suggestion, else null>"
}}"""

NEURO_SYMBOLIC_REASONING_TEMPLATE = """
You are a symbolic reasoning validator for an autonomous agent.
The following symbolic rules have been evaluated against the action plan:

Triggered Violations:
{violations_text}

Action Plan Summary:
{action_summary}

Goal: {primary_goal}
User Constraints: {user_constraints}

Given these violations, assess:
1. Whether the plan is overall safe and feasible
2. Which specific actions need modification
3. What concrete remediation steps are required

Be precise. One paragraph.
"""

MEMORY_RELEVANCE_PROMPT = """
You are a memory retrieval system. Score each memory entry's relevance to the current goal.

Current Goal: {goal}

Memories:
{memories_text}

Return JSON only:
{{
  "scored": [
    {{"id": "<memory_id>", "relevance": <0.0-1.0>, "reason": "<one phrase>"}}
  ]
}}
"""

# ==========================================
# Enums
# ==========================================

class ActionStatus(str, Enum):
    PENDING = "PENDING"
    EXECUTABLE = "EXECUTABLE"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class ConstraintType(str, Enum):
    SAFETY = "SAFETY"
    FEASIBILITY = "FEASIBILITY"
    DEPENDENCY = "DEPENDENCY"
    RESOURCE = "RESOURCE"
    ETHICAL = "ETHICAL"

class EnvironmentType(str, Enum):
    WEB = "WEB"
    FILESYSTEM = "FILESYSTEM"
    API = "API"
    DATABASE = "DATABASE"
    CODE = "CODE"
    GENERAL = "GENERAL"

class IntentType(str, Enum):
    INFORMATION_RETRIEVAL = "INFORMATION_RETRIEVAL"
    TASK_EXECUTION = "TASK_EXECUTION"
    CREATIVE_GENERATION = "CREATIVE_GENERATION"
    ANALYSIS = "ANALYSIS"
    AUTOMATION = "AUTOMATION"
    DECISION_MAKING = "DECISION_MAKING"

class PipelineStatus(str, Enum):
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"

# ==========================================
# Pydantic Models
# ==========================================
# stage 1
class LAMInput(BaseModel):
    """Stage 1 — Validated raw input to the LAM pipeline."""

    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for each request")
    instruction: str = Field(..., description="Natural language action instruction.")
    environment: EnvironmentType = Field(default=EnvironmentType.GENERAL, description="Target execution environment.")
    constraints: list[str] = Field(default_factory=list, description="Hard constraints the action plan must respect.")
    available_tools: list[str] = Field(default_factory=lambda: ["web_search", "code_exec", "file_read", "api_call", "memory_read", "memory_write"], description="Tools available to the action planner.")
    max_steps: int = Field(default=10, ge=1, le=50, description="Maximum number of action steps to plan.")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional contextual information for the agent.")

    @field_validator("instruction")
    @classmethod
    def validate_instruction(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("🚨 Instruction cannot be empty.")
        return stripped

    @model_validator(mode="after")
    def stamp_metadata(self) -> "LAMInput":
        self.metadata["created_at"] = datetime.now(timezone.utc).isoformat()
        return self

# stage 2
class Percept(BaseModel):
    """A single environmental observation"""
    percept_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique identifier for the percept.")
    category: str = Field(..., description="Category of the percept (e.g., 'text', 'image', 'code').")
    observation: str = Field(..., description="Raw observation data.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level of the percept (0.0 to 1.0).")
    source: str = Field(..., description="Source of the percept (e.g., 'web', 'filesystem', 'api').")

class PerceptionResult(BaseModel):
    """Stage 2 - Perception System output"""
    stage: str = "PERCEPTION_SYSTEM"
    environment_type: EnvironmentType = Field(..., description="Type of environment observed.")
    percepts: list[Percept] = Field(default_factory=list, description="List of percepts extracted from the environment.")
    context_summary: str = Field(..., description="Concise summary of the current environment context based on the percepts.")
    complexity_score: float = Field(ge=0.0, le=1.0, description="Estimated complexity of the environment (0.0 to 1.0).")
    processing_time: float = Field(description="Time taken to process the perception stage in seconds.")

# stage 3
class SubGoal(BaseModel):
    """A decomposed sub-goal within the overall intent"""
    goal_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique identifier for the sub-goal.")
    description: str = Field(..., description="Natural language description of the sub-goal.")
    priority: int = Field(ge=1, le=5, description="Priority level of the sub-goal (1 = highest, 5=lowest).")
    measurable: str = Field(..., description="How success of this sub-goal will be measured")

class IntentRecognitionResult(BaseModel):
    """Stage 3 - Intent Recognition output"""
    stage: str = "INTENT_RECOGNITION"
    intent_type: IntentType = Field(..., description="Recognized intent type.")
    primary_goal: str = Field(..., description="Primary goal extracted from the instruction.")
    sub_goals: list[SubGoal] = Field(default_factory=list, description="List of decomposed sub-goals.")
    extracted_entities: list[str] = Field(default_factory=list, description="Key entities extracted from the instruction.")
    temporal_constraint: Optional[str]= Field(default=None, description="Any temporal constraint (deadline) identified in the instruction.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level of the intent recognition (0.0 to 1.0).")
    processing_time: float = Field(description="Time taken to process the intent recognition stage in seconds.")

# stage 4
class AtomicTask(BaseModel):
    """A single atomic task that can be executed independently."""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique identifier for the task.")
    name: str = Field(..., description="Short name of the task (e.g., 'search_web', 'execute_code').")
    description: str = Field(..., description="""Natural language description of the task to be performed.""")
    depends_on: list[str] = Field(default_factory=list, description="List of task_ids that this task depends on.")
    required_tools: list[str] = Field(default_factory=list, description="List of tools required to execute this task.")
    estimated_cost: str = Field(..., description="Estimated cost of executing this task (e.g. low | medium | high).")
    is_critical: bool = Field(default=False, description="Whether this task is critical for achieving the primary goal.")

class TaskBreakdownResult(BaseModel):
    """Stage 4 - Task Breakdown output"""
    stage: str = "TASK_BREAKDOWN"
    tasks: list[AtomicTask] = Field(default_factory=list, description="List of atomic tasks that together achieve the primary goal.")
    task_count: int = Field(description="Total number of atomic tasks generated.")
    critical_path: list[str] = Field(default_factory=list, description="List of task_ids that form the critical path to achieving the primary goal.")
    processing_time: float = Field(description="Time taken to process the task breakdown stage in seconds.")

# stage 5
class Action(BaseModel):
    """A single planned action ready for execution."""
    action_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique identifier for the action.")
    task_id: str = Field(..., description="Identifier of the atomic task this action implements.")
    step: int = Field(..., description="Execution step number for this action.")
    tool: str = Field(..., description="Tool to be used for executing this action.")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameters required for executing the action.")
    expected_output: str = Field(..., description="Expected output or result from executing this action.")
    status: ActionStatus = Field(default=ActionStatus.PENDING, description="Current execution status of the action.")
    rationale: str = Field(..., description="Rationale for why this action is necessary and how it contributes to the overall goal.")
    
class ActionPlanResult(BaseModel):
    """Stage 5 - Action Planning output"""
    stage: str = "ACTION_PLANNING"
    actions: List[Action] = Field(default_factory=list, description="List of planned actions ready for execution.")
    action_count: int = Field(description="Total number of actions in the plan.")
    estimated_total_steps: int = Field(description="Estimated total number of execution steps required to complete the plan.")
    processing_time: float = Field(description="Time taken to process the action planning stage in seconds.")

# stage 6
class EpisodicMemory(BaseModel):
    """A single episodic memory entry."""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique identifier for the memory entry.")
    event: str = Field(..., description="Description of the event or experience to be stored in memory.")
    context: str = Field(..., description="Contextual information surrounding the event.")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat(), description="Timestamp of when the memory was formed.")
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance score of the memory for future retrieval (0.0 to 1.0).")

class SemanticMemory(BaseModel):
    """A single semantic memory entry."""
    fact_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique identifier for the semantic fact.")
    fact: str = Field(..., description="A discrete piece of knowledge or information relevant to the agent's operation.")
    category: str = Field(..., description="Category or type of the semantic fact (e.g., 'tool_capability', 'environment_knowledge').")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence level of the fact's accuracy and relevance (0.0 to 1.0).")

class MemorySystemResult(BaseModel):
    """Stage 6 - Memory System output"""
    stage: str = "MEMORY_SYSTEM"
    episodic_memories: List[EpisodicMemory] = Field(default_factory=list, description="List of episodic memories stored.")
    semantic_memories: List[SemanticMemory] = Field(default_factory=list, description="List of semantic facts stored.")
    relevant_context: str = Field(..., description="Relevant contextual information retrieved from memory to inform action planning.")
    memory_hits: int = Field(description="Number of memory entries retrieved that were relevant to the current task.")
    processing_time: float = Field(description="Time taken to process the memory system stage in seconds.")

# stage 7
class SymbolicConstraint(BaseModel):
    """A symbolic rule applied during neuro-symbolic validation."""
    constraint_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique identifier for the symbolic constraint.")
    constraint_type: ConstraintType = Field(..., description="Type of the constraint (e.g., SAFETY, FEASIBILITY).")
    rule: str = Field(..., description="The symbolic rule or logic that must be satisfied during action execution.")
    applies_to: List[str] = Field(default_factory=list, description="List of action_ids that this constraint applies to.")
    is_satisfied: bool = Field(default=False, description="Whether the constraint is currently satisfied based on the planned actions and memory context.")
    violation_detail: Optional[str] = Field(default=None, description="If the constraint is violated, details about the violation.")

class NeuroSymbolicResult(BaseModel):
    """Stage 7 - Neuro-Symbolic Integration output"""
    stage: str = "NEURO_SYMBOLIC_INTEGRATION"
    constraints_checked: List[SymbolicConstraint] = Field(default_factory=list, description="List of symbolic constraints that were checked against the action plan.")
    violations_found: int = Field(description="Number of symbolic constraint violations found during validation.")
    blocked_actions: List[str] = Field(default_factory=list, description="List of action_ids that are blocked due to constraint violations.")
    approved_actions: List[str] = Field(default_factory=list, description="List of action_ids that are approved for execution after validation.")
    symbolic_reasoning: str = Field(..., description="Explanation of the symbolic reasoning process and how constraints were applied to the action plan.")
    processing_time: float = Field(description="Time taken to process the neuro-symbolic integration stage in seconds.")

# stage 8
class FeedbackEntry(BaseModel):
    """Feedback from a simulated execution step."""
    iteration: int = Field(..., description="Feedback iteration number.")
    action_id: str = Field(..., description="Identifier of the action that was executed.")
    simulated_result: str = Field(..., description="Simulated result of executing the action.")
    success: bool = Field(..., description="Whether the simulated execution was successful.")
    correction: Optional[str] = Field(default=None, description="If the execution failed, a suggested correction or adjustment to the action plan.")

class FeedbackIntegrationResult(BaseModel):
    """Stage 8 - Feedback Integration output"""
    stage: str = "FEEDBACK_INTEGRATION"
    iterations: int = Field(description="Number of feedback iterations completed.")
    feedback_log: List[FeedbackEntry] = Field(default_factory=list, description="Log of feedback entries from each iteration.")
    replanning_count: int = Field(description="Number of times the action plan was replanned based on feedback.")
    final_plan_valid: bool = Field(..., description="Whether the final action plan is considered valid and executable after feedback integration.")
    execution_summary: str = Field(..., description="Summary of the simulated execution process and how feedback was integrated to refine the plan.")
    processing_time: float = Field(description="Time taken to process the feedback integration stage in seconds.")

# stage 9
class LAMOutput(BaseModel):
    """Final structured output of the full LAM pipeline."""
    request_id: str = Field(..., description="Unique identifier for the request, matching the input request_id.")
    stage: str = "OUTPUT"
    status: PipelineStatus = Field(default=PipelineStatus.COMPLETED, description="Overall status of the LAM pipeline execution.")

    # stage payloads
    perception: PerceptionResult
    intent: IntentRecognitionResult
    task_breakdown: TaskBreakdownResult
    action_plan: ActionPlanResult
    memory: MemorySystemResult
    neuro_symbolic: NeuroSymbolicResult
    feedback: FeedbackIntegrationResult

    # final dependencies
    executable_action_plan: List[Action] = Field(default_factory=list, description="List of actions that are approved and executable after the full LAM pipeline processing.")
    final_summary: str = Field(..., description="Concise summary of the entire LAM process, the final action plan, and the expected outcome.")
    total_pipeline_time: float = Field(description="Total time taken to process the entire LAM pipeline in seconds.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata about the LAM execution, such as resource usage, model versions, etc.")

    @model_validator(mode="after")
    def populate_metadata(self) -> "LAMOutput":
        self.metadata['completed_at'] = datetime.now(timezone.utc).isoformat()
        self.metadata['total_actions'] = len(self.executable_action_plan)
        self.metadata['violations'] = self.neuro_symbolic.violations_found
        self.metadata['replanning'] = self.feedback.replanning_count
        
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
        self.logger = get_logger(__name__, log_file="lam.log")
    
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
    
    def _gpt_json_response(self, system: str, user: str, max_tokens: int = 1500) -> dict:
        """wrapper for GPT call with JSON response format."""
        response = self._retry_api_call(
            self.client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        raw = (response.choices[0].message.content or "{}").strip()
        clean = re.sub(r"```(?:json)?|```", "", raw).strip()
        
        try:
            data = json.loads(clean)
            logger.debug(f"💬 parsed json: {data}")
            return data
        except json.JSONDecodeError as e:
            logger.warning(f"JSONDecodeError: {e} | Attempting extraction. Raw: {raw}")
            # Attempt to extract innermost or bounds of { }
            start = clean.find("{")
            end = clean.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    data = json.loads(clean[start:end+1])
                    logger.debug(f"💬 recovered json: {data}")
                    return data
                except json.JSONDecodeError:
                    pass
                    
            logger.error("Failed to recover JSON, returning empty dict to prevent crash.")
            return {}

    def _gpt_text_response(self, system: str, user: str, max_tokens: int = 512) -> str:
        """wrapper for GPT call with plain text response format."""
        response = self._retry_api_call(
            self.client.chat.completions.create,
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        res = (response.choices[0].message.content or "").strip()
        logger.debug(f"💬 raw gpt text response: {res}")
        return res
    
# ==========================================
# Memory Store
# ==========================================
class MemoryStore:
    """
    In-process working memory with episodic (event) and semantic (fact) layers.
    Acts as the LAM's persistent scratchpad across all pipeline stages.
    Uses bounded deques to prevent unbounded memory growth.
    """
    def __init__(self):
        self._episodic_memory: Deque[EpisodicMemory] = deque(maxlen=MEMORY_EPISODIC_LIMIT)
        self._semantic_memory: Deque[SemanticMemory] = deque(maxlen=MEMORY_SEMANTIC_LIMIT)

    def write_episodic(self, event: str, context: str, relevance_score: float = 0.8) -> None:
        """Write a new episodic memory entry."""
        self._episodic_memory.append(EpisodicMemory(event=event, context=context, relevance_score=relevance_score))

    def write_semantic(self, fact: str, category: str, confidence: float = 0.9) -> None:
        """Write a new semantic memory entry."""
        self._semantic_memory.append(SemanticMemory(fact=fact, category=category, confidence=confidence))

    def read_episodic(self, n: int = 10) -> list[EpisodicMemory]:
        """Read episodic memories above a certain relevance threshold."""
        return list(self._episodic_memory)[-n:]
    
    def read_semantic(self, n: int = 20) -> list[SemanticMemory]:
        """Read semantic memories above a certain confidence threshold."""
        return list(self._semantic_memory)[-n:]
    
    def search_episodic(self, keyword: str) -> list[EpisodicMemory]:
        """Search episodic memory for entries containing the keyword."""
        return [mem for mem in self._episodic_memory if keyword.lower() in mem.event.lower() or keyword.lower() in mem.context.lower()]
    
    def search_semantic(self, keyword: str) -> list[SemanticMemory]:
        """Search semantic memory for entries containing the keyword."""
        return [mem for mem in self._semantic_memory if keyword.lower() in mem.fact.lower() or keyword.lower() in mem.category.lower()]
    
    def retrieve_ranked(self, goal: str, llm_call: Callable[[str, str], Dict],
                        top_k: int = 6) -> Tuple[List[EpisodicMemory], List[SemanticMemory]]:
        """
        Score all stored memories against `goal` using a single LLM call,
        then return the top_k most relevant from each layer.
        Falls back gracefully to recency-ordered slice on any error.
        """
        all_ep  = list(self._episodic_memory)
        all_sem = list(self._semantic_memory)

        memories_text = "\n".join(
            f"[EP-{m.memory_id}] event={m.event[:80]}" for m in all_ep
        ) + "\n" + "\n".join(
            f"[SE-{m.fact_id}] fact={m.fact[:80]}" for m in all_sem
        )
        if not memories_text.strip():
            return [], []

        try:
            result = llm_call(
                "You are a memory retrieval system. Score relevance of each memory to the goal. JSON only.",
                MEMORY_RELEVANCE_PROMPT.format(goal=goal, memories_text=memories_text)
            )
            scored: List[Dict] = result.get("scored", [])
            score_map = {s["id"]: float(s.get("relevance", 0.0)) for s in scored}

            def _score_ep(m: EpisodicMemory) -> float:
                return score_map.get(f"EP-{m.memory_id}", m.relevance_score)

            def _score_sem(m: SemanticMemory) -> float:
                return score_map.get(f"SE-{m.fact_id}", m.confidence)

            top_ep = sorted(all_ep, key=_score_ep, reverse=True)[:top_k]
            top_sem = sorted(all_sem,key=_score_sem, reverse=True)[:top_k]
            return top_ep, top_sem

        except Exception as e:
            logger.warning(f"⚠️ Memory ranking failed ({e}), falling back to recency", extra={"tag": "memory"})
            return all_ep[-top_k:], all_sem[-top_k:]

    def log_memory_state(self, logger_instance=None) -> None:
        """Optimally log the current contents of the memory store."""
        logger_instance.info(f"💬 Memory State Snapshot: {len(self._episodic_memory)} Episodic | {len(self._semantic_memory)} Semantic")
        
        if self._episodic_memory:
            logger_instance.debug("--- Episodic Memory ---")
            for mem in self._episodic_memory:
                logger_instance.debug(f"[{mem.relevance_score:.2f}] {mem.event} (Context: {mem.context})")
                
        if self._semantic_memory:
            logger_instance.debug("--- Semantic Memory ---")
            for mem in self._semantic_memory:
                logger_instance.debug(f"[{mem.confidence:.2f} | {mem.category}] {mem.fact}")

# ==========================================
# Pipeline Stages
# ==========================================
class PerceptionSystemStage:
    """
    Stage 2: PERCEPTION SYSTEM
    Senses and classifies the environment and instruction context.
    Produces structured percepts — atomic environmental observations and a complexity score to guide downstream planning.
    """
    def run(self, lam_input: LAMInput, agent: BaseAIAgent) -> PerceptionResult:
        logger.info("⚙️ [PERCEPTION] Sensing environment and context...")
        t0 = time.perf_counter()

        data = agent._gpt_json_response(
            system=(
                "You are a perception system for an autonomous action agent. "
                "Analyse the instruction and environment. Output only valid JSON."
            ),
            user=PERCEPTION_PROMPT_TEMPLATE.format(
                instruction=lam_input.instruction,
                environment=lam_input.environment.value,
                available_tools=", ".join(lam_input.available_tools),
                constraints=", ".join(lam_input.constraints) if lam_input.constraints else "None"
            ),
            max_tokens=1200
        )

        percepts = [Percept(**p) for p in data.get("percepts", [])]
        elapsed = time.perf_counter() - t0
        result = PerceptionResult(
            environment_type=lam_input.environment,
            percepts=percepts,
            context_summary=data.get("context_summary", ""),
            complexity_score=float(data.get("complexity_score", 0.5)),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [PERCEPTION] {len(percepts)} percepts | "
            f"complexity={result.complexity_score:.2f} | time={elapsed:.4f}s"
        )
        logger.debug(f"💬 [perception system stage] | percepts: {percepts} | context_summary: {result.context_summary}")
        return result
    
class IntentRecognitionStage:
    """
    Stage 3: INTENT RECOGNITION
    Extracts the primary goal, decomposed sub-goals, named entities, temporal constraints, and intent type from the perceived context.
    """

    def run(self, lam_input: LAMInput, perception_result: PerceptionResult, agent: BaseAIAgent) -> IntentRecognitionResult:
        logger.info("🎯 [INTENT] Recognizing intent and decomposing goals...")
        t0 = time.perf_counter()

        intent_types = [e.value for e in IntentType]
        data = agent._gpt_json_response(
            system=(
                "You are an intent recognition engine. Extract structured intent "
                "from the user instruction and perceptual context. JSON only."
            ),
            user=INTENT_PROMPT_TEMPLATE.format(
                instruction=lam_input.instruction,
                context_summary=perception_result.context_summary,
                percepts_list=json.dumps([p.observation for p in perception_result.percepts]),
                intent_types=", ".join(intent_types)
            ),
            max_tokens=1200
        )
        elapsed = time.perf_counter() - t0
        result = IntentRecognitionResult(
            intent_type=IntentType(data.get("intent_type", "TASK_EXECUTION")),
            primary_goal=data.get("primary_goal", ""),
            sub_goals=[SubGoal(**sg) for sg in data.get("sub_goals", [])],
            extracted_entities=data.get("extracted_entities", []),
            temporal_constraint=data.get("temporal_constraint"),
            confidence=float(data.get("confidence", 0.8)),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [INTENT] type={result.intent_type.value} | sub_goals={len(result.sub_goals)} | "
            f"confidence={result.confidence:.2f} | time={elapsed:.4f}s"
        )
        logger.debug(f"💬 [intent recognition stage] | primary_goal: {result.primary_goal} | sub_goals: {result.sub_goals} | extracted_entities: {result.extracted_entities} | temporal_constraint: {result.temporal_constraint}")
        return result

class TaskBreakdownStage:
    """
    Stage 4: TASK BREAKDOWN
    Decomposes the recognised intent into a directed task graph of
    atomic, ordered, dependency-linked tasks. Identifies the critical path.
    """
    
    def run(self, lam_input: LAMInput, intent_result: IntentRecognitionResult, agent: BaseAIAgent) -> TaskBreakdownResult:
        logger.info("🛠️ [TASK BREAKDOWN] Decomposing intent into atomic tasks...")
        t0 = time.perf_counter()

        sub_goals_text = "\n".join(
            f"[{sg.goal_id}] {sg.description} (priority={sg.priority})"
            for sg in intent_result.sub_goals
        )
        data = agent._gpt_json_response(
            system=(
                "You are a task decomposition engine. Break down goals into atomic, executable tasks with dependency links. JSON only."
            ),
            user=TASK_BREAKDOWN_PROMPT_TEMPLATE.format(
                primary_goal=intent_result.primary_goal,
                sub_goals_text=sub_goals_text,
                available_tools=", ".join(lam_input.available_tools),
                max_steps=lam_input.max_steps,
                entities=", ".join(intent_result.extracted_entities)
            ),
            max_tokens=1800
        )
        tasks = [AtomicTask(**t) for t in data.get("tasks", [])]
        critical_path = data.get("critical_path", [t.task_id for t in tasks])
        elapsed = time.perf_counter() - t0
        result = TaskBreakdownResult(
            tasks=tasks,
            task_count=len(tasks),
            critical_path=critical_path,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [TASK BREAKDOWN] {len(tasks)} tasks | "
            f"critical_path={len(critical_path)} steps | time={elapsed:.4f}s"
        )
        logger.debug(f"💬 [task breakdown stage] | tasks: {tasks} | critical_path: {critical_path}")
        return result
    
class ActionPlanningStage:
    """
    Stage 5: ACTION PLANNING
    Synthesises a concrete, ordered, tool-grounded action sequence for each task.
    Each action specifies: which tool to call, with what parameters,
    and what output to expect.
    """
    
    def run(self, lam_input: LAMInput, task_breakdown_result: TaskBreakdownResult, intent_result: IntentRecognitionResult, agent: BaseAIAgent, memory_context: str = "") -> ActionPlanResult:
        logger.info(f"🧠 [ACTION PLANNING] Generating actions for {task_breakdown_result.task_count} tasks...")
        t0 = time.perf_counter()
        tasks_text = "\n".join(
            f"[{t.task_id}] {t.name}: {t.description} (tools={t.required_tools}, depends_on={t.depends_on})"
            for t in task_breakdown_result.tasks
        )
        data = agent._gpt_json_response(
            system=(
                "You are an action planning engine for an autonomous agent. "
                "Generate precise, executable actions with tool calls. JSON only."
            ),
            user=ACTION_PLANNING_PROMPT_TEMPLATE.format(
                primary_goal=intent_result.primary_goal,
                tasks_text=tasks_text,
                available_tools=", ".join(lam_input.available_tools),
                constraints=", ".join(lam_input.constraints) if lam_input.constraints else "None",
                memory_context=memory_context or "None"
            ),
            max_tokens=2000
        )
        actions = [Action(**a) for a in data.get("actions", [])]
        elapsed = time.perf_counter() - t0
        result = ActionPlanResult(
            actions=actions,
            action_count=len(actions),
            estimated_total_steps=int(data.get("estimated_total_steps", len(actions))),
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [ACTION PLANNING] {len(actions)} actions generated | "
            f"estimated_total_steps={result.estimated_total_steps} | time={elapsed:.4f}s"
        )
        logger.debug(f"💬 [action planning stage] | actions: {actions}")
        return result

class MemorySystemStage:
    """
    Stage 6: MEMORY SYSTEM
    Populates working memory from the current pipeline state and retrieves
    any contextually relevant memories that could refine the action plan.

    Two memory layers:
    - Episodic: event-tagged memories (what happened, when, in what context)
    - Semantic : fact-tagged memories (world knowledge, constraints, invariants)
    """

    def run(self, lam_input: LAMInput, intent_result: IntentRecognitionResult, action_plan_result: ActionPlanResult, memory_store: MemoryStore, agent: BaseAIAgent) -> MemorySystemResult:
        logger.info("⚙️ [MEMORY SYSTEM] Reading & writing working memory...")
        t0 = time.perf_counter()

        # write current context to memory
        memory_store.write_episodic(
            event=f"New instruction received: {lam_input.instruction[:80]}",
            context=f"Environment: {lam_input.environment.value}",
            relevance_score=0.9
        )
        memory_store.write_episodic(
            event=f"Intent recognized: {intent_result.primary_goal[:80]}",
            context=f"Intent type: {intent_result.intent_type.value}, confidence: {intent_result.confidence:.2f}",
            relevance_score=0.9
        )
        for action in action_plan_result.actions:
            memory_store.write_semantic(
                fact=f"Planned action: {action.tool} ({json.dumps(action.parameters)[:60]})",
                category='action_plan',
                confidence=0.8
            )
        
        # retrieve ranked relevant memories
        top_ep, top_sem = memory_store.retrieve_ranked(
            goal=intent_result.primary_goal,
            llm_call=lambda sys, usr: agent._gpt_json_response(system=sys, user=usr, max_tokens=600),
            top_k=6
        )
        memory_hits = len(top_ep) + len(top_sem)

        # synthesis relevant context
        episodic_text = "\n".join(f" - [{m.timestamp[:19]}] {m.event}" for m in top_ep)
        semantic_text = "\n".join(f" - [{m.category}] {m.fact}" for m in top_sem)

        relevant_context = agent._gpt_text_response(
            system=(
                "You are a memory context synthesiser. Given episodic and semantic memories, produce a single concise paragraph of relevant context that could improve action planning. Be specific."
            ),
            user=(
                f"Goal: {intent_result.primary_goal}\n\n"
                f"Episodic Memories:\n{episodic_text or 'None'}\n\n"
                f"Semantic Memories:\n{semantic_text or 'None'}\n\n"
                "What relevant context from memory should inform the action plan?"
            ),
            max_tokens=300,
        )
        elapsed = time.perf_counter() - t0
        result = MemorySystemResult(
            episodic_memories=top_ep,
            semantic_memories=top_sem,
            relevant_context=relevant_context,
            memory_hits=memory_hits,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [MEMORY SYSTEM] episodic={len(top_ep)} | semantic={len(top_sem)} | hits={memory_hits} | time={elapsed:.4f}s"
        )
        logger.debug(
            f"💬 [memory system stage] | relevant_context: {relevant_context} | "
            f"episodic_memories: {top_ep} | semantic_memories: {top_sem}" 
        )
        return result
    
class NeuroSymbolicIntegrationStage:
    """
    Stage 7: NEURO-SYMBOLIC INTEGRATION
    Validates the action plan through a two-pass hybrid engine:

    Pass 1 — SYMBOLIC (deterministic):
      Hard-coded predicate rules evaluated locally. Non-negotiable.

    Pass 2 — NEURAL (LLM):
      GPT receives the list of *triggered* symbolic violations as hard
      constraints and reasons about whether the plan is overall safe,
      what additional concerns exist, and what remediations are required.
      This is the hybrid integration — both passes inform each other.
    """

    # symbolic rules (deterministic, non-negotiable)
    SYMBOLIC_RULES: List[Dict] = [
        {
            "id"  : "SYM-001",
            "type": ConstraintType.SAFETY,
            "rule": "No action may delete data without an explicit backup step first.",
            "predicate": lambda actions: (
                not any(
                    "delete" in a.tool.lower() or "delete" in str(a.parameters).lower()
                    for a in actions
                )
                or any(
                    "backup" in a.tool.lower() or "backup" in str(a.parameters).lower()
                    for a in actions
                )
            ),
        },
        {
            "id"  : "SYM-002",
            "type": ConstraintType.ETHICAL,
            "rule": "No action may target user PII without explicit consent parameter.",
            "predicate": lambda actions: not any(
                any(kw in str(a.parameters).lower()
                    for kw in ["ssn", "password", "credit_card"])
                and "consent" not in str(a.parameters).lower()
                for a in actions
            ),
        },
        {
            "id"  : "SYM-003",
            "type": ConstraintType.FEASIBILITY,
            "rule": "Total action steps must not exceed MAX_STEPS (50).",
            "predicate": lambda actions: len(actions) <= 50,
        },
        {
            "id"  : "SYM-004",
            "type": ConstraintType.DEPENDENCY,
            "rule": "Every action must reference a valid task_id.",
            "predicate": lambda actions: all(bool(a.task_id) for a in actions),
        },
    ]

    def run(self, lam_input: LAMInput, action_plan_result: ActionPlanResult, intent_result: IntentRecognitionResult, agent: BaseAIAgent) -> NeuroSymbolicResult:
        logger.info(
            f"🛡️ [NEURO-SYMBOLIC] Validating {action_plan_result.action_count} actions against symbolic rules..."
        )
        t0 = time.perf_counter()
        constraints_checked: List[SymbolicConstraint] = []
        blocked_actions: List[str] = []
        violations_found: int = 0
        violation_texts: List[str] = []

        # pass 1: symbolic (deterministic)
        for rule in self.SYMBOLIC_RULES:
            try:
                satisfied = rule["predicate"](action_plan_result.actions)
            except Exception as e:
                agent.logger.error(
                    f"Error evaluating rule {rule['id']}: {e}", extra={"tag": "fail"}
                )
                satisfied = False

            violation_detail: Optional[str] = None
            if not satisfied:
                violations_found += 1
                violation_detail = f'Rule "{rule["rule"]}" was violated.'
                violation_texts.append(f"[{rule['id']} | {rule['type']}] {violation_detail}")
                agent.logger.warning(
                    f"VIOLATION {rule['id']}: {violation_detail}", extra={"tag": "shield"}
                )
                # Hard safety / ethical violations block ALL actions immediately
                if rule["type"] in (ConstraintType.SAFETY, ConstraintType.ETHICAL):
                    blocked_actions = [a.action_id for a in action_plan_result.actions]

            constraints_checked.append(SymbolicConstraint(
                constraint_id=rule["id"],
                constraint_type=rule["type"],
                rule=rule["rule"],
                applies_to=[a.action_id for a in action_plan_result.actions],
                is_satisfied=satisfied,
                violation_detail=violation_detail,
            ))
        
        # pass 2: neural - constrained by symbolic violations
        action_summary = "\n".join(
            f" [{a.action_id}] step {a.step}: {a.tool}({json.dumps(a.parameters)[:60]}) → {a.expected_output}"
            for a in action_plan_result.actions
        )
        user_constraints = "\n".join(f" - {c}" for c in lam_input.constraints) or "None"
        violations_text  = "\n".join(violation_texts) if violation_texts else "None"

        # The LLM receives the fired symbolic violations as hard facts - it cannot override them, only reason about remediation.
        symbolic_reasoning = agent._gpt_text_response(
            system=(
                "You are a neuro-symbolic reasoning validator. "
                "The symbolic rule engine has already flagged violations below — treat them as FACTS. "
                "Your job is to assess overall plan safety, identify any ADDITIONAL concerns the symbolic rules missed, and propose concrete remediations. "
                "One paragraph."
            ),
            user=NEURO_SYMBOLIC_REASONING_TEMPLATE.format(
                violations_text=violations_text,
                action_summary=action_summary,
                primary_goal=intent_result.primary_goal,
                user_constraints=user_constraints,
            ),
            max_tokens=400,
        )

        # compute approved actions
        blocked_set = set(blocked_actions)
        approved_actions = [a.action_id for a in action_plan_result.actions if a.action_id not in blocked_set]

        # update action status
        for action in action_plan_result.actions:
            action.status = (ActionStatus.BLOCKED if action.action_id in blocked_set else ActionStatus.EXECUTABLE)

        elapsed = time.perf_counter() - t0
        result = NeuroSymbolicResult(
            constraints_checked=constraints_checked,
            violations_found=violations_found,
            blocked_actions=blocked_actions,
            approved_actions=approved_actions,
            symbolic_reasoning=symbolic_reasoning,
            processing_time=round(elapsed, 4)
        )
        logger.info(
            f"✅ [NEURO-SYMBOLIC] checked={len(constraints_checked)} | "
            f"violations={violations_found} | blocked={len(blocked_actions)} | approved={len(approved_actions)} | time={elapsed:.4f}s"
        )
        logger.debug(f"💬 [neuro-symbolic stage] | symbolic_reasoning: {symbolic_reasoning}")
        return result
    
class FeedbackIntegrationStage:
    """
    Stage 8: FEEDBACK INTEGRATION
    Simulates execution of the approved action plan, collects feedback
    on each step, and triggers re-planning if failures are detected.

    This implements the LAM's core adaptive loop:
    This implements the true adaptive loop: Plan → Sim → Eval → Re-plan (if needed) → Re-sim.
    """
    def run(self, lam_input: LAMInput,
            neuro_symbolic: NeuroSymbolicResult,
            action_plan: ActionPlanResult,
            intent: IntentRecognitionResult,
            memory_store: MemoryStore,
            agent: BaseAIAgent
        ) -> FeedbackIntegrationResult:

        logger.info(
            f"Simulating {len(neuro_symbolic.approved_actions)} approved actions…",
            extra={"tag": "feedback"}
        )
        t0 = time.perf_counter()

        approved_ids = set(neuro_symbolic.approved_actions)
        active_actions = [a for a in action_plan.actions if a.action_id in approved_ids]
        feedback_log: List[FeedbackEntry] = []
        replanning_count = 0

        def _simulate(action: Action, iteration: int) -> FeedbackEntry:
            data = agent._gpt_json_response(
                system="You are an action execution simulator. JSON only.",
                user=FEEDBACK_PROMPT_TEMPLATE.format(
                    primary_goal=intent.primary_goal,
                    tool=action.tool,
                    parameters=json.dumps(action.parameters),
                    expected_output=action.expected_output,
                    rationale=action.rationale,
                ),
                max_tokens=1500,
            )
            raw_sim = data.get("simulated_result", "")
            sim_str = raw_sim if isinstance(raw_sim, str) else json.dumps(raw_sim)
            return FeedbackEntry(
                iteration=iteration,
                action_id=action.action_id,
                simulated_result=sim_str,
                success=bool(data.get("success", True)),
                correction=data.get("correction"),
            )

        iteration = 1
        for action in active_actions:
            entry = _simulate(action, iteration)
            feedback_log.append(entry)
            iteration += 1

            if entry.success:
                action.status = ActionStatus.COMPLETED
                memory_store.write_episodic(
                    event=f"Action {action.action_id} completed: {action.tool}",
                    context=f"Result: {entry.simulated_result[:100]}",
                    relevance_score=0.7,
                )
            else:
                action.status = ActionStatus.FAILED
                replanning_count += 1
                logger.warning(
                    f"⚠️ Action {action.action_id} failed. Correction: {entry.correction}",
                    extra={"tag": "adapt"}
                )

                if replanning_count >= MAX_FEEDBACK_ITERATIONS:
                    logger.error(
                        f"❌ Max re-plan iterations ({MAX_FEEDBACK_ITERATIONS}) reached. Halting.",
                        extra={"tag": "fail"}
                    )
                    break

                # re-plan for this failed action 
                logger.info(
                    f"⚠️ Re-planning failed action {action.action_id}…",
                    extra={"tag": "adapt"}
                )
                failed_text = (
                    f"[{action.action_id}] step {action.step}: "
                    f"{action.tool}({json.dumps(action.parameters)}) → "
                    f"FAILED. Correction hint: {entry.correction or 'none'}"
                )
                corrections = entry.correction or "Try alternative approach."

                try:
                    replan_data = agent._gpt_json_response(
                        system=(
                            "You are an action re-planner. Generate replacement actions ONLY for the failed action. JSON only."
                        ),
                        user=ACTION_REPLAN_PROMPT_TEMPLATE.format(
                            primary_goal=intent.primary_goal,
                            failed_actions=failed_text,
                            memory_context=memory_store.retrieve_ranked(
                                intent.primary_goal,
                                lambda s, u: agent._gpt_json_response(s, u, 400),
                                top_k=3,
                            )[0], # just episodic list; str conversion below
                            constraints=", ".join(lam_input.constraints) or "None",
                            corrections=corrections,
                        ),
                        max_tokens=800,
                    )
                    replacement_actions = [
                        Action(**a) for a in replan_data.get("actions", [])
                    ]
                    if replacement_actions:
                        # Replace failed action with first replacement; simulate
                        replacement = replacement_actions[0]
                        replacement.status = ActionStatus.PENDING
                        # Update in action_plan for final output
                        for i, a in enumerate(action_plan.actions):
                            if a.action_id == action.action_id:
                                action_plan.actions[i] = replacement
                                break
                        re_entry = _simulate(replacement, iteration)
                        feedback_log.append(re_entry)
                        iteration += 1
                        if re_entry.success:
                            replacement.status = ActionStatus.COMPLETED
                            logger.info(
                                f"✅ Re-planned action {replacement.action_id} succeeded",
                                extra={"tag": "success"}
                            )
                        else:
                            replacement.status = ActionStatus.FAILED
                            logger.warning(f"⚠️ Re-planned action {replacement.action_id} also failed. Correction: {re_entry.correction}", extra={"tag": "warn"})
                except Exception as e:
                    logger.error(f"❌ Error during re-planning: {e}", extra={"tag": "fail"})

        final_plan_valid = (
            replanning_count < MAX_FEEDBACK_ITERATIONS
            and all(e.success for e in feedback_log)
        )
        successes = sum(1 for e in feedback_log if e.success)

        execution_summary = agent._gpt_text_response(
            system="You are an execution reporter. Summarise what was accomplished. 2 sentences.",
            user=(
                f"Goal: {intent.primary_goal}\n"
                f"Actions Simulated: {len(feedback_log)}\n"
                f"Successes: {successes} | Failures: {len(feedback_log) - successes}\n"
                f"Re-planning cycles: {replanning_count}\n"
            ),
            max_tokens=200,
        )
        elapsed = time.perf_counter() - t0

        result = FeedbackIntegrationResult(
            iterations=iteration - 1,
            feedback_log=feedback_log,
            replanning_count=replanning_count,
            final_plan_valid=final_plan_valid,
            execution_summary=execution_summary,
            processing_time=round(elapsed, 4),
        )
        logger.info(
            f"✅ [FEEDBACK] iterations={len(feedback_log)} | "
            f"replanning_count={replanning_count} | final_plan_valid={final_plan_valid} | time={elapsed:.4f}s"
        )
        logger.debug(f"💬 [feedback integration stage] | feedback_log: {feedback_log} | execution_summary: {execution_summary}")
        return result
                
# ==========================================
# LAM Agent - orchestrates all 9 pipeline stages
# ==========================================
class LAMAgent(BaseAIAgent):
    """
    Large Action Model Agent (LAMAgent). 
    Core principle: operates in ACTION SPACE — plans, validates, adapts, and remembers, not just generates.
    """
    def __init__(self, client: OpenAI) -> None:
        super().__init__(client)
        self._perception = PerceptionSystemStage()
        self._intent = IntentRecognitionStage()
        self._task_breakdown = TaskBreakdownStage()
        self._action_plan = ActionPlanningStage()
        self._memory_stage = MemorySystemStage()
        self._neuro_symbolic = NeuroSymbolicIntegrationStage()
        self._feedback = FeedbackIntegrationStage()
        self._memory_store = MemoryStore()
        self._registered_tools: List[str] = []  

    def register_tool(self, tool_name: str, description: str = "") -> None:
        """Register a custom tool name so it appears in planning prompts."""
        self._registered_tools.append(tool_name)
        self.logger.info(
            f"Tool '{tool_name}' registered ({description})",
            extra={"tag": "tool"}
        )

    def process(self, lam_input: LAMInput) -> LAMOutput:
        """
        Execute the full 9-stage LAM pipeline with typed checkpointing.
        Resume is automatic — if a checkpoint exists for request_id, completed stages are skipped.
        """
        pipeline_start = time.perf_counter()
        rid = lam_input.request_id

        self.logger.info(
            f"Pipeline START | request_id={rid[:8]} | model={CHAT_MODEL}",
            extra={"tag": "boot"}
        )

        # Merge registered tools into available_tools
        if self._registered_tools:
            lam_input.available_tools = list(
                dict.fromkeys(lam_input.available_tools + self._registered_tools)
            )

        try:
            # ── Stage 2: Perception ───────────────────────────────────────────
            perception = checkpointer.load("perception", PerceptionResult)
            if perception is None:
                perception = self._perception.run(lam_input, self)
                checkpointer.save("perception", perception)

            # ── Stage 3: Intent Recognition ───────────────────────────────────
            intent = checkpointer.load("intent", IntentRecognitionResult)
            if intent is None:
                intent = self._intent.run(lam_input, perception, self)
                checkpointer.save("intent", intent)

            # ── Stage 4: Task Breakdown ───────────────────────────────────────
            task_breakdown = checkpointer.load("task_breakdown", TaskBreakdownResult)
            if task_breakdown is None:
                task_breakdown = self._task_breakdown.run(lam_input, intent, self)
                checkpointer.save("task_breakdown", task_breakdown)

            # ── Stage 5 (pre-memory): Initial Action Plan ────────────────────
            #    We plan first without memory context, then enrich memory,
            #    then re-plan if memory adds significant context.
            action_plan = checkpointer.load("action_plan", ActionPlanResult)
            if action_plan is None:
                action_plan = self._action_plan.run(
                    lam_input, task_breakdown, intent, self, memory_context=""
                )
                checkpointer.save("action_plan", action_plan)

            # ── Stage 6: Memory System ←→ Stage 5 feedback ───────────────────
            memory = checkpointer.load("memory", MemorySystemResult)
            if memory is None:
                memory = self._memory_stage.run(
                    lam_input, intent, action_plan, self._memory_store, self
                )
                checkpointer.save("memory", memory)
                # Re-plan enriched with memory context (bidirectional loop)
                if memory.memory_hits > 0 and memory.relevant_context.strip():
                    self.logger.info(
                        "Memory context available — re-enriching action plan…",
                        extra={"tag": "adapt"}
                    )
                    action_plan = self._action_plan.run(
                        lam_input, task_breakdown, intent, self,
                        memory_context=memory.relevant_context
                    )
                    checkpointer.save("action_plan", action_plan)

            # ── Stage 7: Neuro-Symbolic Integration ───────────────────────────
            neuro_symbolic = checkpointer.load("neuro_symbolic", NeuroSymbolicResult)
            if neuro_symbolic is None:
                neuro_symbolic = self._neuro_symbolic.run(
                    lam_input, action_plan, intent, self
                )
                checkpointer.save("neuro_symbolic", neuro_symbolic)

            # ── Stage 8: Feedback Integration ─────────────────────────────────
            feedback = checkpointer.load("feedback", FeedbackIntegrationResult)
            if feedback is None:
                feedback = self._feedback.run(
                    lam_input, neuro_symbolic, action_plan,
                    intent, self._memory_store, self
                )
                checkpointer.save("feedback", feedback)

            # ── Stage 9: Output ───────────────────────────────────────────────
            final_summary = checkpointer.load_raw_key("final_summary")
            if final_summary is None:
                final_summary = self._gpt_text_response(
                    system="You are an action plan reporter. Be concise and precise.",
                    user=(
                        f"Goal: {intent.primary_goal}\n"
                        f"Approved: {len(neuro_symbolic.approved_actions)}\n"
                        f"Violations: {neuro_symbolic.violations_found}\n"
                        f"Execution Valid: {feedback.final_plan_valid}\n"
                        f"Execution Summary: {feedback.execution_summary}\n\n"
                        "Write a final 2-sentence action plan summary."
                    ),
                    max_tokens=200,
                )
                checkpointer.save_raw_key("final_summary", final_summary)

            executable_actions = [
                a for a in action_plan.actions
                if a.status in (ActionStatus.EXECUTABLE, ActionStatus.COMPLETED)
            ]

            # FIX-09: status reflects true pipeline outcome
            if not feedback.final_plan_valid:
                pipeline_status = PipelineStatus.PARTIAL
            else:
                pipeline_status = PipelineStatus.COMPLETED

            total_time = time.perf_counter() - pipeline_start
            output = LAMOutput(
                request_id=rid,
                status=pipeline_status,
                perception=perception,
                intent=intent,
                task_breakdown=task_breakdown,
                action_plan=action_plan,
                memory=memory,
                neuro_symbolic=neuro_symbolic,
                feedback=feedback,
                executable_action_plan=executable_actions,
                final_summary=final_summary,
                total_pipeline_time=round(total_time, 4),
                metadata={
                    **lam_input.metadata,
                    "model": CHAT_MODEL,
                }
            )
            # clear checkpoint on success if desired: 
            # if checkpoint_file and os.path.exists(checkpoint_file): os.remove(checkpoint_file)

            self.logger.info(
                f"🎉 [LAM AGENT] Pipeline COMPLETE | total_time={total_time:.4f}s | "
                f"tasks={task_breakdown.task_count} | actions={action_plan.action_count} | "
                f"approved={len(neuro_symbolic.approved_actions)} | violations={neuro_symbolic.violations_found}"
            )
            self._memory_store.log_memory_state(self.logger)
            return output
        except Exception as e:
            elapsed = time.perf_counter() - pipeline_start
            self.logger.error(
                f"❌ [LAM AGENT] Pipeline FAILED after {elapsed:.4f}s | "
                f"error={type(e).__name__}: {e}"
            )
            raise

    def display_output(self, output: LAMOutput) -> None:
        DIV  = "═" * 100

        print(f"\n{DIV}")
        print("🟡 LAM AGENT — Large Action Model Pipeline Result")
        print(DIV)

        print(f"Request ID: {output.request_id}")
        print(f"Total Time: {output.total_pipeline_time:.2f}s")
        print(f"Model: {output.metadata.get('model', CHAT_MODEL)}")

        print(DIV)
        print("📝 FINAL SUMMARY")
        print(DIV)
        for line in textwrap.wrap(output.final_summary, width=120):
            print(f"{line}")
        print(f"\n{DIV}")
        
        print("🛠️ SIMULATED ACTION RESULTS")
        print(DIV)
        for entry in output.feedback.feedback_log:
            print(f"\n▶ Action [{entry.action_id}] Success: {entry.success}")
            print("  Result:")
            print(textwrap.indent(entry.simulated_result, "    "))
            
        print(f"\n{DIV}\n")

# ==========================================
# Instatiation
# ==========================================
def create_lam_agent(api_key: str = TOKEN, endpoint: str = ENDPOINT) -> LAMAgent:
    """Factory function to create an instance of LAMAgent with the provided API credentials."""
    client = OpenAI(
        base_url=endpoint,
        api_key=api_key,
    )
    logger.info("🏭 [FACTORY] LAMAgent instantiated and ready.")
    return LAMAgent(client)

# ==========================================
# Example Usage
# ==========================================

if __name__ == "__main__":
    # create agent instance
    agent = create_lam_agent(TOKEN, ENDPOINT)

    # Optional: register a custom tool 
    agent.register_tool("markdown_writer", "Writes structured markdown reports to disk")

    # Build input
    lam_input = LAMInput(
        instruction=(
            "Research the top open-source LLM frameworks available in 2026, then generate a structured markdown report."
        ),
        environment=EnvironmentType.WEB,
        constraints=[
            "Do not access any paywalled sources.",
            "Report must be under 500 words.",
        ],
        available_tools=[
            "web_search", "web_fetch", "markdown_writer", "file_write", "memory_read", "memory_write",
        ],
        max_steps=15,
        metadata={"source": "lam_agent_demo", "version": "1.0"},
    )

    # Run pipeline 
    result = agent.process(lam_input)

    # Display results 
    agent.display_output(result)
