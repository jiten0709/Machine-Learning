<h1 align="center">🧠 Specialized AI Agents</h1>
A modular, production-grade repository showcasing distinct AI agent architectures. Built with a focus on type safety, observability, and deterministic stage execution.

## Table of Contents

- Technical Stack
- Project Overview
- Quickstart
- Agent: LLM (Large Language Model)
- Agent: LCM (Large Concept Model)
- Agent: LAM (Large Action Model)
- Logging & Observability
- Contributing
- Security & Secrets

## ⚙️ Technical Stack

- **Core**: Python 3.11+, Pydantic v2, ABC (Abstract Base Classes)
- **AI/ML**: OpenAI API (gpt-4o, text-embedding-3-small), NLTK, tiktoken
- **Infra**: `uv` package manager, `.env` for secrets management

## 🏗 Project Overview

This architecture treats AI pipelines as strict, sequential data contracts. Each stage is a discrete, validated component.

Design guarantees:

- **Type Safety**: Pydantic models validate all inputs and outputs between pipeline stages.
- **Resiliency**: Exponential backoff and retry logic for API calls.
- **Modularity**: Abstract base classes (`BaseAIAgent`) enforce a uniform interface.
- **Observability**: Structured, per-stage logging with execution timings.

## 🚀 Quickstart

**Prerequisites:** Python 3.11+

**Installation:**

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Configuration:**

```bash
cp .env.example .env
# Edit .env with your GITHUB_TOKEN and GITHUB_ENDPOINT
```

**Execution:**

```bash
python3 LLM.py
```

## 🔵 Agent 1: LLM (Large Language Model)

A strict-pipeline conversational agent.

**Pipeline Flow:**

```mermaid
graph LR
    A[Input] --> B[Tokenization]
    B --> C[Embedding]
    C --> D[Transformer]
    D --> E[Output]
```

| Stage            | Implementation                                    | Data Contract       |
| :--------------- | :------------------------------------------------ | :------------------ |
| **Input**        | Raw text ingestion & validation                   | `LLMInput`          |
| **Tokenization** | Real token counting & truncation via `tiktoken`   | `TokenizedResult`   |
| **Embedding**    | Semantic vectorization (`text-embedding-3-small`) | `EmbeddingResult`   |
| **Transformer**  | Attention-based generation (`gpt-4o` family)      | `TransformerResult` |
| **Output**       | Aggregated pipeline metadata and final text       | `LLMOutput`         |

## 🟢 Agent 2: LCM (Large Concept Model)

An advanced agent operating in latent concept space rather than surface token space.

**Pipeline Flow:**

```mermaid
graph LR
    A[Input] --> B[Sentence Segmentation]
    B --> C[SONAR Embedding]
    C --> D[Diffusion]
    D --> E[Advanced Patterning]
    D --> F[Hidden Process]
    E --> G[Quantization]
    F --> G
    G --> H[Output]
```

| Stage               | Implementation                                     | Data Contract              |
| :------------------ | :------------------------------------------------- | :------------------------- |
| **Segmentation**    | Atomic unit splitting via NLTK `punkt`             | `SegmentationResult`       |
| **SONAR Embedding** | Per-sentence OpenAI embeddings + mean pooling      | `SonarEmbeddingResult`     |
| **Diffusion**       | DDPM-inspired Gaussian noise refinement            | `DiffusionResult`          |
| **Patterning**      | Structural maps via abstract prompt instruction    | `AdvancedPatterningResult` |
| **Hidden Process**  | Cosine similarity clustering & cross-inference     | `HiddenProcessResult`      |
| **Quantization**    | 8-bit uniform scalar quantization for state limits | `QuantizationResult`       |
| **Output**          | Full state aggregation and structural summary      | `LCMOutput`                |

## 🟠 Agent 3: LAM (Large Action Model)

An "executive function" agent that operates in _action space_ rather than token space. It handles complex, multi-step tasks requiring dynamic planning, persistent memory, and deterministic symbolic reasoning.

**Pipeline Flow:**

```mermaid
graph LR
    A[Input] --> B[Perception]
    B --> C[Intent Recognition]
    C --> D[Task Breakdown]
    D --> E[Action Planning]
    E <--> F[Memory System]
    E <--> G[Neuro-Symbolic]
    F <--> G
    G --> H[Feedback Integration]
    H --> I[Output]
```

| Stage                  | Implementation                                         | Data Contract               |
| :--------------------- | :----------------------------------------------------- | :-------------------------- |
| **Input**              | Natural language instruction & environment config      | `LAMInput`                  |
| **Perception**         | Environmental observation & complexity scoring         | `PerceptionResult`          |
| **Intent Recognition** | Goal extraction & sub-goal decomposition               | `IntentRecognitionResult`   |
| **Task Breakdown**     | Directed atomic task graph & critical path computation | `TaskBreakdownResult`       |
| **Action Planning**    | Tool-grounded, sequential action synthesis             | `ActionPlanResult`          |
| **Memory System**      | Episodic (events) & semantic (facts) working memory    | `MemorySystemResult`        |
| **Neuro-Symbolic**     | Neural checking + deterministic safety rule predicates | `NeuroSymbolicResult`       |
| **Feedback**           | Execution simulation & adaptive replanning loops       | `FeedbackIntegrationResult` |
| **Output**             | Executable action plan aggregation & final summary     | `LAMOutput`                 |

## 📊 Logging & Observability

- Centralized logger configured via `logging_setup.py`.
- Execution timings are tracked at the nanosecond level per stage using `time.perf_counter()`.
- Output is written locally to `logs/<agent_name>.log`.

## 🤝 Contributing

- Open an issue for architectural proposals.
- Maintain backward compatibility with the `BaseAIAgent` abstract interface.
- Ensure strict Pydantic v2 schema adherence for new data contracts.

## 🔒 Security & Secrets

- Do not commit `.env` files.
- Logs are sanitized locally. Ensure `system_prompt` and Pydantic object dumps do not expose user PII.

<p align="center">Made with ❤️ by Jiten.</p>
