<h1 align="center">Computer Vision</h1>

<p align="center">
  Seven production-grade PyTorch notebooks, from nine multiply-adds in a sliding window
  to a dense per-pixel classifier with a verified deployment artefact.
</p>

<p align="center">
  <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.6%2B-ee4c2c?logo=pytorch&logoColor=white">
  <img alt="Python" src="https://img.shields.io/badge/Python-3.10%2B-3776ab?logo=python&logoColor=white">
  <img alt="Albumentations" src="https://img.shields.io/badge/Albumentations-2.0%2B-8a2be2">
  <img alt="ONNX" src="https://img.shields.io/badge/ONNX-Runtime-005ce6?logo=onnx&logoColor=white">
  <img alt="Devices" src="https://img.shields.io/badge/device-CUDA%20%7C%20MPS%20%7C%20CPU-success">
</p>

---

## What this is

A deliberately-ordered curriculum, not a folder of experiments. Each notebook answers one question,
ends with an exported artefact that has been **numerically verified against eager PyTorch**, and
hands off to the next.

Every dataset downloads automatically. Every notebook runs on CUDA, Apple MPS, or plain CPU.

| #   | Notebook                                                                            | Question answered                                        | Dataset                 |
| :-- | :---------------------------------------------------------------------------------- | :------------------------------------------------------- | :---------------------- |
| 01  | [Convolution & Pooling from Scratch](01_convolution_and_pooling_from_scratch.ipynb) | What _is_ a convolution — and can the kernel be learned? | `scikit-image` samples  |
| 02  | [Image Classification: MLP vs CNN](02_image_classification_mlp_vs_cnn.ipynb)        | What is the convolutional prior actually worth?          | Fashion-MNIST           |
| 03  | [Regularisation & Augmentation](03_regularization_and_augmentation.ipynb)           | How do you stop a model memorising its training set?     | CIFAR-10                |
| 04  | [Transfer Learning (ResNet-50)](04_transfer_learning_resnet50.ipynb)                | Why start from random weights at all?                    | Oxford-IIIT Pet         |
| 05  | [Vision Transformer Fine-Tuning](05_vision_transformer_finetuning.ipynb)            | What happens if you remove the spatial prior?            | Oxford-IIIT Pet         |
| 06  | [Object Detection (Faster R-CNN)](06_object_detection_faster_rcnn.ipynb)            | What, and **where**?                                     | Penn-Fudan              |
| 07  | [Semantic Segmentation (U-Net)](07_semantic_segmentation_unet.ipynb)                | What, for **every pixel**?                               | Oxford-IIIT Pet trimaps |

---

## Quickstart

```sh
python -m venv .venv && source .venv/bin/activate     # macOS / Linux
pip install -r vision/requirements.txt
jupyter lab vision/
```

**CUDA users** — install torch first from the index matching your driver, then the requirements file:

```sh
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r vision/requirements.txt
```

### These notebooks ship tuned for **speed**, not for peak accuracy

They are reference material — every notebook's _default_ configuration is its fastest one, sized to
finish in minutes rather than hours. Every reduced value is a single named constant (`epochs`,
`TRAIN_SUBSET_SIZE`, `TEST_SUBSET_SIZE`, …) with the production value noted in a comment beside it —
raise the constant, nothing else needs to change.

| Notebook | Ships with (default)                   | Raise to, for a trustworthy result                           |
| :------- | :------------------------------------- | :----------------------------------------------------------- |
| 02       | 1 epoch, 4,000 train / 2,000 test      | 15 epochs                                                    |
| 03       | 2 epochs, 1,000 train / 1,000 test     | 12 epochs (visible gap) → 40+ (full separation)              |
| 04       | 1 epoch, 370 train / 370 test          | 4 epochs (already decisive) → 8–15, `*_SUBSET_SIZE = None`   |
| 05       | 1 epoch, same subsets as 04            | 3 epochs → 6–10, `*_SUBSET_SIZE = None`                      |
| 06       | 1 epoch, 16 train / 8 test             | 4 epochs (clear of baseline) → 10–20, `*_SUBSET_SIZE = None` |
| 07       | 2 epochs, 200 train / 100 test, 128 px | 6 epochs (clear skip/no-skip gap) → 12–30, 256 px            |

Each notebook also carries a **Production note** callout at the point where the fast default first
matters (right after training), spelling out exactly what to raise and why.

### One environment switch

```sh
DETERMINISTIC=0 jupyter lab vision/     # let cuDNN autotune — big win for conv-heavy notebooks
```

This is the first thing to try if a notebook feels slow on a CUDA GPU. The default sets
`cudnn.deterministic = True` and `cudnn.benchmark = False`, which affect **convolutions only** — so
the symptom is a CNN crawling while an MLP in the same notebook runs fine. Turning it off trades exact
run-to-run reproducibility for cuDNN's algorithm autotuner.

Worker counts are capped at `os.cpu_count()` rather than hardcoded, because augmentation runs on the
CPU and a 2-vCPU host (Colab free tier) will starve the GPU no matter how fast the GPU is.

---

## The seven-section structure

Every notebook follows the same skeleton, so once you have read one you can navigate any of them.

| §   | Section                       | Contents                                                                                 |
| :-- | :---------------------------- | :--------------------------------------------------------------------------------------- |
| 1   | **Header & Context**          | Overview, LaTeX formulation, ASCII architecture diagram, provenance                      |
| 2   | **Setup & Reproducibility**   | Seeded RNGs, `cuda → mps → cpu` detection, printed version table                         |
| 3   | **Data Pipeline**             | Albumentations pipelines, typed `Dataset`, a _visual_ check of the augmentations         |
| 4   | **Modular Architecture**      | Frozen-dataclass configs, `nn.Module` classes, Google-style docstrings, shape assertions |
| 5   | **Production Training Loop**  | AMP, gradient clipping, warmup + cosine, per-epoch logging, best-checkpointing           |
| 6   | **Inference & Visualisation** | Curves, confusion matrices, bbox overlays, mask overlays, attention maps                 |
| 7   | **Deployment Readiness**      | TorchScript + ONNX export, **parity assertion**, CPU latency benchmark                   |

### Engineering conventions

- **PEP 8**, explicit `typing` hints on every signature, Google-style docstrings with `Shape:` blocks.
- **No magic numbers.** Every hyper-parameter lives in a `@dataclass(frozen=True)` — printable,
  diffable, and overridable in one place.
- **Markdown before code**, explaining _why_ — the loss choice, the LR schedule shape, the
  augmentation strength, and in several places why a tempting option was **rejected**.
- **Modern PyTorch.** `torch.amp` rather than the deprecated `torch.cuda.amp`; mixed precision is
  gated per device (fp16 + `GradScaler` on pre-Ampere CUDA, bf16 on Ampere+/CPU, off on MPS where
  autocast coverage is still partial).
- **Self-contained.** No shared utility module to import — each notebook opens and runs anywhere,
  including Colab and Kaggle. The six duplicated lines of seed/device setup are a deliberate trade
  for that portability.

---

## Datasets and artefacts live under the OS temp directory, not the repo

Every notebook writes downloads to `{tempfile.gettempdir()}/vision-data` and checkpoints/exported
models to `{tempfile.gettempdir()}/vision-artifacts/<notebook>` — `/tmp/...` on Colab and Linux, an
OS-managed cache directory on macOS. Nothing is created inside the `vision/` folder itself. This
repo assumes ephemeral execution (Colab first and foremost): there is no persistent local checkout to
keep clean, and re-running a notebook in a fresh Colab session starts from a clean slate either way.

Within a single running session, re-executing cells reuses whatever was already downloaded or
trained (the temp path is fixed, not randomised per run) — only a new machine or a wiped `/tmp`
starts over.

| Dataset                |    Size | Used by    | Note                                         |
| :--------------------- | ------: | :--------- | :------------------------------------------- |
| Fashion-MNIST          |   30 MB | 02         | Fast                                         |
| CIFAR-10               |  170 MB | 03         | Fast                                         |
| Penn-Fudan             |   53 MB | 06         | Fast                                         |
| Oxford-IIIT Pet        | ~800 MB | 04, 05, 07 | ⚠️ **Slow.** See below                       |
| `scikit-image` samples |       0 | 01         | Ships inside the package — no network at all |

> **⚠️ Oxford-IIIT Pet downloads from `robots.ox.ac.uk`, which is frequently very slow** — measured at
> ~70 KB/s during development, i.e. **around three hours** for the full dataset. Notebooks 04, 05 and
> 07 subset it heavily by default (down to 370 / 370 images and smaller — see the table above), which
> cuts the download to a fraction of that; only raising `*_SUBSET_SIZE` toward `None` for a production
> run brings the full three-hour cost back. Start it in the background before you need it:
>
> ```sh
> python -c "
> import tempfile
> from pathlib import Path
> from torchvision.datasets import OxfordIIITPet
> root = Path(tempfile.gettempdir()) / 'vision-data'
> for s in ('trainval', 'test'):
>     OxfordIIITPet(root=root, split=s, target_types=['category', 'segmentation'], download=True)"
> ```

Pretrained weights (ResNet-50 ≈ 100 MB, ViT-B/16 ≈ 330 MB, Faster R-CNN ≈ 170 MB) come from
`download.pytorch.org`, which is CDN-backed and fast, and are cached by `torch.hub` in its own
standard location — untouched by anything above.

---

<h3 align="center">Made with ❤️ by Jiten.</h3>
