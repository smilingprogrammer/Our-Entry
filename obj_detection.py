from pathlib import Path
from typing import List

import cv2
import mindspore_lite as mslite
import numpy as np
import pyttsx3

# Paths
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "tinyvit_glasses.mindir"
LABELS_PATH = ROOT / "labels.txt"

# Image + normalization params (must match training pipeline)
IMG_SIZE = 32
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR_STD = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

# Runtime knobs
CAMERA_INDEX = 0
FRAME_SKIP = 10
CONFIDENCE_THRESHOLD = 0.7


def load_labels(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text().splitlines()]
    labels = [line for line in lines if line]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits)
    exp = np.exp(logits)
    return exp / np.sum(exp)


def preprocess(frame: np.ndarray) -> np.ndarray:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
    normalized = resized.astype(np.float32) / 255.0
    normalized = (normalized - CIFAR_MEAN) / CIFAR_STD
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0).copy()


def build_ms_lite_model(model_path: Path) -> mslite.Model:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing MindSpore Lite model: {model_path}. "
            f"Convert the MindIR exported in huawei-futa.ipynb to .mslite first."
        )

    context = mslite.Context()
    cpu_info = mslite.CPUDeviceInfo(enable_fp16=False, thread_num=2)
    context.append_device_info(cpu_info)

    model = mslite.Model()
    model.build_from_file(str(model_path), mslite.ModelType.MSLITE_MINDIR, context)
    return model


def run():
    labels = load_labels(LABELS_PATH)
    model = build_ms_lite_model(MODEL_PATH)
    input_tensor = model.get_inputs()[0]
    output_tensor = model.get_outputs()[0]

    engine = pyttsx3.init()
    last_label = ""
    frame_count = 0

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Camera not accessible")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_array = preprocess(frame)
            input_tensor.set_data_from_numpy(input_array)
            model.run()
            logits = np.squeeze(output_tensor.get_data_to_numpy())
            probs = softmax(logits)

            label_id = int(np.argmax(probs))
            confidence = float(probs[label_id])
            label = labels[label_id]

            cv2.putText(
                frame,
                f"{label} ({confidence:.2f})",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("TinyViT MindSpore Lite", frame)

            if frame_count % FRAME_SKIP == 0 and confidence > CONFIDENCE_THRESHOLD and label != last_label:
                engine.say(f"{label} detected")
                engine.runAndWait()
                last_label = label

            frame_count += 1
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
