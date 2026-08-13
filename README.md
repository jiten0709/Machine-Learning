<h1 align="center">Machine Learning</h1>

<p align="center">
  A practical workshop for ML and applied AI — classical algorithms, a PyTorch computer-vision
  curriculum, an LLM fine-tuning stack, and engineered agent architectures.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Jupyter" src="https://img.shields.io/badge/Jupyter-notebooks-f37626?logo=jupyter&logoColor=white">
</p>

---

## 📌 Overview

Four self-contained tracks, each with its own README, dependencies, and quickstart — there is no
shared package or install; pick a directory and follow its instructions. [`csv/`](csv/) covers
classical ML on tabular data (regression, classification, PCA, clustering, ensembles).
[`vision/`](vision/) is a seven-notebook PyTorch computer-vision curriculum from convolution
fundamentals through object detection and segmentation, each ending in a verified ONNX export.
[`llm-fine-tuning/`](llm-fine-tuning/) is a 24-notebook curriculum covering SFT, the PEFT/LoRA
family, preference alignment (DPO/PPO and friends), and the distributed tooling that runs them.
[`specialized-ai-models/`](specialized-ai-models/) implements eight agent architectures (LLM, LCM,
LAM, MoE, VLM, SLM, MLM, SAM) as production-style Pydantic-validated pipelines rather than prompt
scripts.

## 📁 Repository Structure

```
Machine-Learning/
├── csv/                      # Classical ML notebooks on tabular datasets
│   └── assets/                 # CSV datasets used by the notebooks
├── vision/                   # PyTorch CV curriculum (7 notebooks)
│   ├── data/                   # Auto-downloaded datasets
│   └── artifacts/               # Exported ONNX/TorchScript models
├── llm-fine-tuning/           # LLM adaptation curriculum (24 notebooks)
│   ├── supervised-fine-tuning/
│   ├── peft/
│   ├── alignment-techniques-and-preference-optimization/
│   ├── advanced-fine-tuning-paradigms/
│   ├── domain-specific/
│   └── tools-and-frameworks/
├── specialized-ai-models/     # 8 engineered agent pipelines (LLM/LCM/LAM/MoE/VLM/SLM/MLM/SAM)
│   └── utils/                   # Shared logging + checkpointing
└── Notes/                     # Scratch notes
```

## 🚀 Getting Started

There is no root-level dependency file — each track manages its own environment because their
dependencies conflict (different `torch`/CUDA builds, GPU-only libraries like `bitsandbytes` and
Unsloth, an API-key-based agent runtime). Clone the repo, then set up whichever track you need:

```sh
git clone https://github.com/jiten0709/Machine-Learning.git
cd Machine-Learning
```

| Track                                              | Setup                                                                                                                                                                                    |
| -------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`csv/`](csv/)                                     | `pip install jupyter pandas scikit-learn xgboost matplotlib seaborn`, then `jupyter lab csv/`                                                                                            |
| [`vision/`](vision/)                               | `pip install -r vision/requirements.txt`, then `jupyter lab vision/` — see [`vision/README.md`](vision/README.md)                                                                        |
| [`llm-fine-tuning/`](llm-fine-tuning/)             | Per-notebook installs (`transformers`, `peft`, `trl`, plus the framework each notebook needs) — see [`llm-fine-tuning/README.md`](llm-fine-tuning/README.md)                             |
| [`specialized-ai-models/`](specialized-ai-models/) | `pip install -r specialized-ai-models/requirements.txt`, copy `.env.example` to `.env` and set `GITHUB_TOKEN` — see [`specialized-ai-models/README.md`](specialized-ai-models/README.md) |

## Contributing

- Open issues for new concepts or fixes.
- Add notebooks or scripts with a clear README and reproducible steps.
- Keep experiments deterministic and document dependencies in the track they belong to.

---

<h3 align="center">Made with ❤️ by Jiten.</h3>
