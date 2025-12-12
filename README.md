# TinyViT on CIFAR-10 (MindSpore)

## Overview
- Train lightweight Vision Transformers (TinyViT) on CIFAR-10 while keeping compute needs small.
- Notebook ports the full workflow from PyTorch to MindSpore: data prep, model, training loops, and random-search HPO.
- Produces checkpoints, JSON logs, and plots for both the baseline run and tuned trials.

## Requirements
| Component | Version/Notes |
| --- | --- |
| Python | 3.9 or newer |
| MindSpore | 2.2+ (GPU build preferred, CPU fallback supported) |
| CUDA/cuDNN | Match the MindSpore GPU wheel (omit if CPU only) |
| Python packages | numpy, pandas, matplotlib |
| Hardware | GPU with >=8 GB VRAM recommended; notebook falls back to CPU |

## Setup
1. Create and activate a virtual environment.
2. Install MindSpore (GPU example):
   ```bash
   pip install mindspore-gpu==2.2.0 -i https://pypi.mindspore.cn/simple
   ```
   CPU alternative:
   ```bash
   pip install mindspore==2.2.0
   ```
3. Install helpers:
   ```bash
   pip install numpy pandas matplotlib
   ```
4. Launch Jupyter or VS Code and open `tiny-vit-cifar-10.ipynb`.

## Notebook Flow
1. **Background & Motivation** – links to TinyViT / ViT resources.
2. **Imports & Context** – sets MindSpore seeds and device target (tries GPU, falls back to CPU).
3. **Data Pipeline** – MindSpore `Cifar10Dataset` loaders with RandAugment, optional CutOut, and a validation split.
4. **Model** – TinyViT `nn.Cell` with custom multi-head attention, GELU MLP head, class token, and positional embeddings.
5. **Training Utilities** – MixUp helper, label smoothing, `ops.value_and_grad` training loop, MindSpore validation routine.
6. **Baseline Run** – 1-layer TinyViT (patch=4, emb=64) for 20 epochs; logs metrics to `baseline_results/`.
7. **Hyperparameter Search** – random sampling over patch size, embedding dim, MLP multiplier, heads, dropout, LR, weight decay, batch size, MixUp alpha, label smoothing, RandAugment, CutOut.
8. **Visualization & Summaries** – accuracy plots plus JSON exports for downstream analysis.

## Running the Baseline
1. Execute cells through the "Baseline TinyViT (1 layer) run" section.
2. Generated artifacts:
   - `baseline_best_checkpoint.ckpt` – best validation checkpoint.
   - `baseline_results/tinyvit_baseline_final.ckpt` – final weights.
   - `baseline_results/history.json`, `baseline_results/summary.json`, `baseline_results/baseline_acc_plot.png`.
3. To restart cleanly, delete or rename `baseline_results/` or adjust `baseline_config`.

## Random Search Workflow
1. Set `num_trials` (default 8, comment suggests 32) and adjust `sample_random_config()` as needed.
2. Run the hyperparameter-optimization cells.
3. Outputs include:
   - `random_search_results/all_results.json` – metrics for every trial.
   - `random_search_results/best_tinyvit_randomsearch.json` – best trial summary.
   - `random_search_results/best_val_acc_per_trial.png` and `random_search_results/best_trial_learning_curves.png`.
4. Console logs show each trial's config, per-epoch metrics, total training time, inference time, and the `(patch, layers, emb_dim, mlp_dim, heads)` tuple.

## Customization Tips
- Increase `emb_dim`, `heads`, or add more encoder layers for stronger models (watch VRAM).
- Toggle augmentations (`randaugment`, `cutout`, MixUp alpha) to match your dataset.
- Replace CIFAR-10 by editing `get_cifar10_loaders_with_val` to load a custom MindSpore dataset.
- Save extra checkpoints inside `run_one_trial` if you want per-trial weights.

## Troubleshooting
| Issue | Fix |
| --- | --- |
| MindSpore wheel fails | Verify Python and CUDA versions, then install the matching wheel from the MindSpore site. |
| CIFAR-10 download blocked | Manually download to `./data` and rerun loaders with `download=False`. |
| GPU OOM | Lower `batch_size`, disable MixUp/CutOut, or switch to CPU mode. |
| Slow CPU run | Reduce `max_epochs` or `num_trials`, or run on GPU/Ascend hardware. |

## References
- TinyViT paper (Han et al., 2022) and related posts cited in the notebook intro.
- ReadyTensor CIFAR-10 ViT tutorial.
- DMICZ "minViT" notes.
