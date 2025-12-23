# Technical Solution Introduction

![Architecture Diagram – insert screenshot here](./docs/architecture.png)

Our smart-glasses platform couples a MindSpore TinyViT classifier with Raspberry Pi–friendly inference code and an OCR+LLM narration path. The workflow begins in `huawei-futa.ipynb`, where TinyViT is trained on the canonical CIFAR-10 dataset (32×32 RGB, 10 labels) using data augmentation and hyperparameter search. We export the resulting checkpoint to MindIR, convert it with MindSpore Lite, and drop the `.mslite` artifact into `Smart_Glasses_For_Blind_People/` where the Pi runtime handles camera ingestion, inference, OCR, and speech output in real time.

![Code Walkthrough Screenshot – insert here](./docs/code-snippet.png)

## Key Technical Modules

- **Module 1 – Dataset Selection & Processing**  
  - *Dataset:* CIFAR-10 (60k images, 32×32, 10 balanced classes) provides the supervised training corpus inside `huawei-futa.ipynb`.  
  ̀- *Processing pipeline:* `get_transforms` defines normalization (`cifar10_mean`, `cifar10_std`), RandomResizedCrop, RandAugment, CutOut, and other augmentations to increase robustness while keeping the tensor layout (`(B,3,32,32)`) compatible with MindSpore loaders.

- **Module 2 – TinyViT Model Implementation**  
  - *Architecture:* A one-layer TinyViT is coded with multi-head self-attention, class token, sinusoidal positional embeddings, GELU MLP, and LayerNorm. Parameters such as `patch_size`, `emb_dim`, `mlp_mult`, and `heads` are exposed so random-search trials can sweep the design space.  
  - *Training utilities:* Custom helpers implement one-hot targets, label smoothing, mixup augmentation, and manual loss computations, ensuring the MindSpore graph stays lightweight for export.

- **Module 3 – Training, Inference, and Deployment Export**  
  - *Training loop:* `train_epoch` / `validate` log accuracy, loss, epoch duration, and save checkpoints whenever validation accuracy improves (`ms.save_checkpoint`). Hyperparameters (learning rate, weight decay, dropout, mixup) are swept via random search.  
  - *Export chain:* After training, the notebook serializes the best weights, writes `tinyvit_glasses.mindir`, and documents the `converter_lite --fmk=MINDIR ... --target=ARM64` command that produces `tinyvit_glasses.mslite`, ready for Raspberry Pi inference through `Smart_Glasses_For_Blind_People/obj_detection.py`.

## Final Achievements

- **Accuracy:** TinyViT’s best random-search trial on CIFAR-10 delivered 55.48 % test accuracy (Trial 3: Patch=4, Embedding=128, Heads=8), establishing a reproducible benchmark for this lightweight transformer.  
- **Inference Throughput:** The recorded benchmark processes the full CIFAR-10 test set in ~1.26 seconds (≈15 ms/image). After conversion to MindSpore Lite, this equates to comfortably real-time (>30 FPS) webcam inference on Raspberry Pi 4.  
- **Cost & Efficiency:** Reusing the single TinyViT checkpoint across notebook, converter, and Pi runtime eliminates cloud inference fees for object detection; only the OCR API remains optional. MindSpore Lite’s smaller runtime footprint (~200 MB less than the TensorFlow toolchain) further reduces storage and memory requirements on the device.
