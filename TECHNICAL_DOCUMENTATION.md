# Smart Glasses for the Visually Impaired – Technical Documentation

## 1. Project Overview
- **Problem:** Blind users need lightweight, offline assistance to recognize nearby objects and read printed text.
- **Solution:** Train a TinyViT classifier (MindSpore) on CIFAR-10, convert it to MindSpore Lite for Raspberry Pi inference, and pair it with an OCR+Gemini narration pipeline.
- **Key Features:** Real-time object detection with spoken feedback, OCR capture with LLM summarization, fully local inference for detection, modular Raspberry Pi deployment.

## 2. Architecture
1. **Training Layer (PC/GPU):** `huawei-futa.ipynb` handles CIFAR-10 loading, TinyViT training, checkpointing, and MindIR export.
2. **Conversion Layer (PC):** MindSpore Lite `converter_lite` transforms `tinyvit_glasses.mindir` into `tinyvit_glasses.mslite`.
3. **Edge Layer (Raspberry Pi 4):**
   - `obj_detection.py` → MindSpore Lite runtime + pyttsx3 speech.
   - `ocr.py` → OpenCV capture + OCR API + Gemini summarization.
4. **Artifacts:** `labels.txt`, `tinyvit_glasses.mslite`, runtime scripts.

*(Insert architecture diagram here)*

## 3. Setup Instructions
1. **Prerequisites**
   - Python 3.10+, MindSpore 2.0 (GPU build recommended).
   - MindSpore Lite converter binaries (host) + runtime package (ARM64).
   - Raspberry Pi 4 with Raspberry Pi OS 64-bit.
2. **Training Environment**
   ```bash
   pip install mindspore-gpu==2.0.* pandas matplotlib numpy
   ```
   - Open `huawei-futa.ipynb` and run baseline + random search cells.
3. **Export & Convert**
   - Run the notebook cell that exports `Smart_Glasses_For_Blind_People/tinyvit_glasses.mindir`.
   - Convert to a lightweight version of mindspore:
     ```bash
     converter_lite \
       --fmk=MINDIR \
       --modelFile=Smart_Glasses_For_Blind_People/tinyvit_glasses.mindir \
       --outputFile=Smart_Glasses_For_Blind_People/tinyvit_glasses \
       --target=ARM64 \
       --optimize=ascend_oriented
     ```
4. **Raspberry Pi Setup**
   ```bash
   pip install -r Smart_Glasses_For_Blind_People/requirements.txt
   ```
   - Install MindSpore Lite runtime wheel/binaries for ARM64.
   - Set environment variables for OCR/Gemini keys inside `ocr.py`.

## 4. Usage Guide
- **Object Detection**
  ```bash
  python Smart_Glasses_For_Blind_People/obj_detection.py
  ```
  - Press `q` to exit. Requires camera on `/dev/video0`.
- **OCR Narration**
  ```bash
  python Smart_Glasses_For_Blind_People/ocr.py
  ```
  - Takes a 3-second preview, captures a frame, hits OCR API, summarizes with Gemini.
- **Notebook Experiments**
  - Use `run_one_trial` to benchmark new hyperparameters; results printed and plotted inside the notebook.

## 5. Implementation Details
- **Dataset Pipeline:** CIFAR-10 via MindSpore Dataset API with shared transforms (RandomResizedCrop, RandAugment, CutOut, normalization). Ensures identical preprocessing between training and deployment.
- **Model:** Custom TinyViT (1 transformer layer) with parameterizable `patch_size`, `emb_dim`, `heads`. Includes custom multi-head attention and MLP built in MindSpore for portability.
- **Training Logic:** Manual mixup, label smoothing, and AdamW optimizer. Validation-driven checkpointing, history logging, and Matplotlib plots for accuracy curves.
- **Inference Runtime:** MindSpore Lite Python API replaces TensorFlow Lite. Frames are normalized with CIFAR stats, fed to `tinyvit_glasses.mslite`, and top-1 label is spoken when confidence >0.7.
- **OCR Flow:** OpenCV capture → multipart POST to OCR API → Gemini 2.0 Flash summarization → offline TTS playback.

## 6. Testing
- **Notebook Validation:** `validate` function computes accuracy on CIFAR-10 validation/test splits each epoch; best checkpoint chosen by validation accuracy.
- **Manual Runtime Tests:** On Raspberry Pi, run `obj_detection.py` with sample objects to confirm FPS and spoken output; log console confidence values.
- **Planned Enhancements:** Add pytest module for preprocessing + label synchronization and integrate automated smoke tests for MindSpore Lite loading (not yet implemented).

## 7. Deployment
1. Copy `Smart_Glasses_For_Blind_People/` (including `.mslite`, `labels.txt`, Python scripts) to the Pi.
2. Install dependencies via requirements file and MindSpore Lite runtime.
3. Enable camera (`sudo raspi-config` → Interface Options → Camera).
4. Launch `obj_detection.py` as a systemd service or via `tmux` for continuous use; optionally bind hotkeys to toggle OCR.

## 8. Challenges & Solutions
| Challenge                                       | Solution                                                                                 |
|-------------------------------------------------|------------------------------------------------------------------------------------------|
| Limited Pi resources for transformers           | Adopted TinyViT (single-layer) and MindSpore Lite, yielding a small `.mslite` artifact. |
| Keeping labels consistent between notebook & Pi | Notebook writes `Smart_Glasses_For_Blind_People/labels.txt` after training/export.       |
| Dependency on cloud inference                   | Object detection fully offline; only OCR uses paid API, with clear configuration hooks. |
| Real-time speech jitter                         | Added frame skipping + confidence gating (0.7 threshold) to reduce repeated announcements. |

## 9. Future Work
1. Fine-tune TinyViT directly on a custom smart-glasses dataset (replace CIFAR-10).
2. Introduce temporal smoothing or multi-label support for crowded scenes.
3. Replace paid OCR API with an efficient on-device OCR (e.g., PaddleOCR).
4. Add automated integration tests for MindSpore Lite inference on the Pi.
5. Build a lightweight mobile companion app for remote monitoring and updates.

---
This document summarizes the technical design, setup, and operational playbook for the smart-glasses solution. Judges and collaborators can follow these sections to reproduce training, conversion, and deployment end-to-end.
