#Read raw MNIST data, without dataset specific library

from pathlib import Path
import numpy as np


def load_images(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 16 or int.from_bytes(raw[:4], "big") != 2051:
        raise ValueError(f"Bad IDX image header: {path}")
    count = int.from_bytes(raw[4:8], "big")
    rows = int.from_bytes(raw[8:12], "big")
    cols = int.from_bytes(raw[12:16], "big")
    if (rows, cols) != (28, 28) or len(raw) != 16 + count * rows * cols:
        raise ValueError(f"Bad MNIST image dimensions/length: {path}")
    
    # Scale grayscale pixel bytes from [0, 255] to [0, 1]

    return np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(count, 784).astype(np.float32) / 255.0


def load_labels(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    if len(raw) < 8 or int.from_bytes(raw[:4], "big") != 2049:
        raise ValueError(f"Bad IDX label header: {path}")
    count = int.from_bytes(raw[4:8], "big")
    if len(raw) != 8 + count:
        raise ValueError(f"Bad MNIST label length: {path}")
    labels = np.frombuffer(raw, dtype=np.uint8, offset=8).copy()
    if np.any(labels > 9):
        raise ValueError("MNIST labels must be between 0 and 9")
    return labels


def load_split(folder: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    prefix = {"train": "train", "test": "t10k"}[split]
    images = load_images(folder / f"{prefix}-images-idx3-ubyte")
    labels = load_labels(folder / f"{prefix}-labels-idx1-ubyte")
    if len(images) != len(labels):
        raise ValueError("Image/label count mismatch")
    return images, labels
