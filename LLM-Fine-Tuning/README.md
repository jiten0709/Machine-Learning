<h1 align="center">LLM Fine-Tuning</h1>

<p align="center">
  The collection of notebooks covering the full LLM adaptation stack — supervised
  fine-tuning, parameter-efficient methods, preference alignment, and the distributed tooling
  that runs them at scale.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Transformers" src="https://img.shields.io/badge/%F0%9F%A4%97-Transformers%20%7C%20PEFT%20%7C%20TRL-yellow">
</p>

---

## 📌 Overview

This directory is a curriculum, not a scratch folder: each notebook isolates one fine-tuning
technique — what problem it solves, the mechanics under the hood, and a runnable implementation
against a real model and dataset. Coverage spans four layers of the stack: adapting a base model's
behavior (SFT, continued pre-training, multi-task and instruction tuning), reducing the trainable
parameter footprint (LoRA family, BitFit, DoRA, prompt/prefix tuning), aligning outputs to human or
AI-generated preferences (DPO, IPO, KTO, ORPO, SimPO, RLHF/RLAIF with PPO), and the frameworks that
make training tractable on real hardware (Axolotl, Unsloth, LLaMA-Factory, DeepSpeed vs. FSDP). A
`domain-specific/` track applies the same techniques to a concrete pharma use case, with its own
instruction and preference data.

## ✨ Key Features

- Full PEFT family — LoRA, QLoRA, AdaLoRA, DoRA, BitFit, prompt/prefix tuning — each isolated in
  its own notebook for direct comparison.
- Reward-based and reward-free alignment methods (PPO-based RLHF/RLAIF alongside DPO, IPO, KTO,
  ORPO, SimPO) implemented against the same preference-data pattern.
- Distributed training tooling notebook contrasting DeepSpeed ZeRO 1–3 sharding against PyTorch
  FSDP.
- Framework-level walkthroughs for Axolotl (declarative YAML configs), Unsloth (custom Triton
  kernels), and LLaMA-Factory.
- End-to-end domain-specific example: pharma-focused instruction and preference tuning with its
  own CSV/PDF data assets.

## 🛠️ Tech Stack & Dependencies

| Category         | Tools                                                                   |
| ---------------- | ----------------------------------------------------------------------- |
| Language         | Python 3.10+                                                            |
| Core ML          | PyTorch, Hugging Face `transformers`, `datasets`, `safetensors`         |
| Fine-tuning      | `peft`, `trl`, `bitsandbytes`                                           |
| Frameworks       | Axolotl, Unsloth, LLaMA-Factory, DeepSpeed                              |
| Notebook / utils | Jupyter, `pandas`, `matplotlib`, `pymupdf` (`fitz`), `pyyaml`, `psutil` |

No `requirements.txt` is checked in for this directory — each tool (Axolotl, Unsloth, LLaMA-Factory,
DeepSpeed, `bitsandbytes`) has its own install path and GPU/CUDA constraints, and mixing them into
one environment is not recommended. Install what a given notebook needs from its first cell.

## 📁 Project Structure

```
llm-fine-tuning/
├── supervised-fine-tuning/              # CLM and MLM-style SFT
├── peft/
│   ├── additive-methods/                # LoRA, QLoRA, AdaLoRA, prompt/prefix tuning
│   └── selective-reparameterization-methods/  # BitFit, DoRA
├── alignment-techniques-and-preference-optimization/
│   ├── direct-reward-free-alignment-methods/  # DPO, IPO, KTO, ORPO, SimPO
│   └── reward-based-methods/            # RLHF / RLAIF with PPO
├── advanced-fine-tuning-paradigms/      # CPT, instruction tuning, MTFT, data packing
├── domain-specific/                     # Pharma instruction + preference tuning example
│   └── assets/                          # Source PDF, instruction/preference CSVs
└── tools-and-frameworks/                # Axolotl, Unsloth, LLaMA-Factory, DeepSpeed vs FSDP
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- A CUDA-capable GPU for anything beyond the smallest PEFT runs (`bitsandbytes` quantization,
  DeepSpeed/FSDP sharding, and Unsloth's Triton kernels are CUDA-only)
- Jupyter (Lab or Notebook)

### Installation

Clone the repo:

```sh
git clone https://github.com/jiten0709/Machine-Learning.git
cd Machine-Learning/llm-fine-tuning
```

Create and activate a virtual environment:

```sh
python -m venv .venv
source .venv/bin/activate
```

Install the core stack shared by most notebooks:

```sh
pip install torch transformers datasets peft trl bitsandbytes accelerate
```

Install a framework only when its notebook needs it (each pins its own versions in-notebook):

```sh
pip install axolotl        # tools-and-frameworks/axolotl_declarative_tuning.ipynb
pip install unsloth         # tools-and-frameworks/unsloth_fast_kernel_tuning.ipynb
pip install llamafactory    # tools-and-frameworks/llama_factory.ipynb
pip install deepspeed       # tools-and-frameworks/distributed_sharding_deepspeed_vs_fsdp.ipynb
```

### Environment Variables

None are required to run the notebooks locally. If a notebook pulls a gated model or pushes to the
Hugging Face Hub, set:

| Variable   | Description                                                        |
| ---------- | ------------------------------------------------------------------ |
| `HF_TOKEN` | Hugging Face access token, for gated models or `push_to_hub` calls |

## 🏃 Usage & Running

Launch Jupyter from this directory and open any notebook directly:

```sh
jupyter lab .
```

There is no shared CLI, build, or test/lint tooling in this directory — each notebook is
self-contained and runs top-to-bottom.

## 🔌 Notebook Reference

| Track           | Notebook                                                                                                                                                    | Covers                                     |
| --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ |
| SFT             | [`supervised-fine-tuning/sft_with_clm.ipynb`](supervised-fine-tuning/sft_with_clm.ipynb)                                                                    | Causal-LM supervised fine-tuning           |
| SFT             | [`supervised-fine-tuning/sft_with_mlm.ipynb`](supervised-fine-tuning/sft_with_mlm.ipynb)                                                                    | Masked-LM supervised fine-tuning           |
| PEFT            | [`peft/additive-methods/LoRA.ipynb`](peft/additive-methods/LoRA.ipynb)                                                                                      | Low-Rank Adaptation                        |
| PEFT            | [`peft/additive-methods/QLoRA.ipynb`](peft/additive-methods/QLoRA.ipynb)                                                                                    | 4-bit quantized LoRA                       |
| PEFT            | [`peft/additive-methods/AdaLoRA.ipynb`](peft/additive-methods/AdaLoRA.ipynb)                                                                                | Adaptive rank allocation                   |
| PEFT            | [`peft/additive-methods/prompt_and_prefix_tuning.ipynb`](peft/additive-methods/prompt_and_prefix_tuning.ipynb)                                              | Soft prompt / prefix tuning                |
| PEFT            | [`peft/selective-reparameterization-methods/BitFit.ipynb`](peft/selective-reparameterization-methods/BitFit.ipynb)                                          | Bias-only fine-tuning                      |
| PEFT            | [`peft/selective-reparameterization-methods/DoRA.ipynb`](peft/selective-reparameterization-methods/DoRA.ipynb)                                              | Weight-decomposed LoRA                     |
| Alignment       | [`.../direct-reward-free-alignment-methods/DPO.ipynb`](alignment-techniques-and-preference-optimization/direct-reward-free-alignment-methods/DPO.ipynb)     | Direct Preference Optimization             |
| Alignment       | [`.../direct-reward-free-alignment-methods/IPO.ipynb`](alignment-techniques-and-preference-optimization/direct-reward-free-alignment-methods/IPO.ipynb)     | Identity Preference Optimization           |
| Alignment       | [`.../direct-reward-free-alignment-methods/KTO.ipynb`](alignment-techniques-and-preference-optimization/direct-reward-free-alignment-methods/KTO.ipynb)     | Kahneman-Tversky Optimization              |
| Alignment       | [`.../direct-reward-free-alignment-methods/ORPO.ipynb`](alignment-techniques-and-preference-optimization/direct-reward-free-alignment-methods/ORPO.ipynb)   | Odds Ratio Preference Optimization         |
| Alignment       | [`.../direct-reward-free-alignment-methods/SimPO.ipynb`](alignment-techniques-and-preference-optimization/direct-reward-free-alignment-methods/SimPO.ipynb) | Simple Preference Optimization             |
| Alignment       | [`.../reward-based-methods/RLHF_with_PPO.ipynb`](alignment-techniques-and-preference-optimization/reward-based-methods/RLHF_with_PPO.ipynb)                 | RLHF with PPO                              |
| Alignment       | [`.../reward-based-methods/RLAIF_with_PPO.ipynb`](alignment-techniques-and-preference-optimization/reward-based-methods/RLAIF_with_PPO.ipynb)               | RLAIF with PPO                             |
| Advanced        | [`advanced-fine-tuning-paradigms/cpt.ipynb`](advanced-fine-tuning-paradigms/cpt.ipynb)                                                                      | Continued pre-training / domain adaptation |
| Advanced        | [`advanced-fine-tuning-paradigms/instruction_tuning.ipynb`](advanced-fine-tuning-paradigms/instruction_tuning.ipynb)                                        | Instruction tuning                         |
| Advanced        | [`advanced-fine-tuning-paradigms/mtft.ipynb`](advanced-fine-tuning-paradigms/mtft.ipynb)                                                                    | Multi-task fine-tuning                     |
| Advanced        | [`advanced-fine-tuning-paradigms/data_packing_strategies.ipynb`](advanced-fine-tuning-paradigms/data_packing_strategies.ipynb)                              | Sequence packing strategies                |
| Domain-specific | [`domain-specific/instruction_pretrain_finetuning.ipynb`](domain-specific/instruction_pretrain_finetuning.ipynb)                                            | Pharma instruction-tuned pretraining       |
| Domain-specific | [`domain-specific/non_instruction_pretrain_finetuning.ipynb`](domain-specific/non_instruction_pretrain_finetuning.ipynb)                                    | Pharma non-instruction pretraining         |
| Domain-specific | [`domain-specific/preference-based/DPO.ipynb`](domain-specific/preference-based/DPO.ipynb)                                                                  | Pharma preference-based DPO                |
| Tools           | [`tools-and-frameworks/axolotl_declarative_tuning.ipynb`](tools-and-frameworks/axolotl_declarative_tuning.ipynb)                                            | Declarative fine-tuning with Axolotl       |
| Tools           | [`tools-and-frameworks/unsloth_fast_kernel_tuning.ipynb`](tools-and-frameworks/unsloth_fast_kernel_tuning.ipynb)                                            | Custom Triton kernels with Unsloth         |
| Tools           | [`tools-and-frameworks/llama_factory.ipynb`](tools-and-frameworks/llama_factory.ipynb)                                                                      | LLaMA-Factory pipeline                     |
| Tools           | [`tools-and-frameworks/distributed_sharding_deepspeed_vs_fsdp.ipynb`](tools-and-frameworks/distributed_sharding_deepspeed_vs_fsdp.ipynb)                    | DeepSpeed ZeRO 1–3 vs. PyTorch FSDP        |

---

<h3 align="center">Made with ❤️ by Jiten.</h3>
