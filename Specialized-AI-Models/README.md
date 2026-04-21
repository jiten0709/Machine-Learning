# Specialized AI Agents

A compact, modular repository showcasing 8 different specialized AI agents. Each component is designed to be replaceable, observable, and testable.

## Table of Contents

- Project Overview
- Quickstart
- Agent: LLM (Large Language Model)
- Files and Structure
- Logging & Observability
- Contributing

## Project Overview

This project demonstrates a clear, stage-oriented LLM pipeline where each stage is a discrete unit (ingest, tokenization, embedding, transformer, output). The architecture emphasizes validation, logging, and the ability to swap implementations (mock vs. production).

## Quickstart

Requirements:

- Python 3.11+
- Virtual environment recommended

Install:

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Setup necessary environment variables:

```
create .env file and refer .env.example
```

Run a quick example (replace with actual runner or script):

```bash
python3 LLM.py
```

## Architecture Overview

Design principles:

- Each pipeline stage is a first-class, logged, validated component.
- Pydantic models enforce input/output schemas.
- Swap in real providers easily (OpenAI, local models, or mocked functions).
- Fine-grained logging for observability and debugging.

## Agent: LLM — Large Language Model

🔵 Agent 1: LLM — Large Language Model

📐 Architecture Breakdown

Based on the project diagram, the LLM pipeline follows this strict sequential flow:
Input → Tokenization → Embedding → Transformer → Output

| Stage        |                                    Role | Implementation Strategy          |
| ------------ | --------------------------------------: | -------------------------------- |
| Input        |         Raw text ingestion & validation | Pydantic model with guardrails   |
| Tokenization | Simulate token count, chunking strategy | tiktoken for real token counting |
| Embedding    |          Semantic vector representation | OpenAI text-embedding-3-small    |
| Transformer  |  Attention-based reasoning & generation | OpenAI gpt-4.1                   |
| Output       |          Structured, validated response | Pydantic output model + metadata |

## Logging & Observability

- Centralized logger configured in logging_setup.py.

Logs are written to logs/\<filename>.log by default.

## Contributing

- Open an issue for major changes or architecture proposals.
- Small bugfixes and docs improvements: open a PR with a short description and tests if applicable.
- Maintain backward-compatible changes for adapter interfaces.

## Security & Secrets

- Do not commit API keys or secrets. Use .env and .env.example for guidance.
- Sanitize logs: never log raw secrets or full user inputs in plaintext.

# Made with ❤️ by Jiten.
