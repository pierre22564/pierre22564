from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


IMAGE_METHODS = {"gaussian_blur"}

EMBEDDING_METHODS = {
    "embedding_noise",
    "embedding_dp_laplace",
    "random_projection",
    "quantization",
    "cancellable_transform",
}


@dataclass(frozen=True)
class MethodSpec:
    name: str
    space: str
    description: str
    default_params: dict[str, Any]


METHOD_SPECS: dict[str, MethodSpec] = {
    "gaussian_blur": MethodSpec(
        "gaussian_blur",
        "image",
        "Gaussian blur applied to the input image.",
        {"kernel_size": 9},
    ),
    "embedding_noise": MethodSpec(
        "embedding_noise",
        "embedding",
        "Gaussian noise directly on the embedding.",
        {"sigma": 0.05},
    ),
    "embedding_dp_laplace": MethodSpec(
        "embedding_dp_laplace",
        "embedding",
        "Laplace noise on the embedding controlled by epsilon.",
        {"epsilon": 8.0, "sensitivity": 1.0},
    ),
    "random_projection": MethodSpec(
        "random_projection",
        "embedding",
        "Johnson-Lindenstrauss style random projection.",
        {"target_dim": 128, "seed": 42},
    ),
    "quantization": MethodSpec(
        "quantization",
        "embedding",
        "Uniform scalar quantization.",
        {"levels": 32},
    ),
    "cancellable_transform": MethodSpec(
        "cancellable_transform",
        "embedding",
        "Keyed permutation and sign flip transform.",
        {"seed": 42, "mix_ratio": 1.0},
    ),
}


def available_methods() -> list[str]:
    return list(METHOD_SPECS.keys())


def build_visual_embedding_map(embedding: np.ndarray) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    padded = np.zeros(529, dtype=np.float32)
    padded[: min(len(vector), 529)] = vector[:529]
    heatmap = padded.reshape(23, 23)
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-12)
    return heatmap


def apply_image_method(image_rgb: np.ndarray, method: str, params: dict[str, Any]) -> np.ndarray:
    image = image_rgb.copy()

    if method == "gaussian_blur":
        kernel_size = _odd_kernel(int(params.get("kernel_size", 9)))
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    raise ValueError(f"Unsupported image method: {method}")


def apply_embedding_method(embedding: np.ndarray, method: str, params: dict[str, Any]) -> np.ndarray:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)

    if method == "embedding_noise":
        sigma = float(params.get("sigma", 0.05))
        return _l2_normalize(vector + np.random.normal(0.0, sigma, vector.shape).astype(np.float32))

    if method == "embedding_dp_laplace":
        epsilon = max(float(params.get("epsilon", 8.0)), 1e-6)
        sensitivity = float(params.get("sensitivity", 1.0))
        scale = sensitivity / epsilon
        return _l2_normalize(vector + np.random.laplace(0.0, scale, vector.shape).astype(np.float32))

    if method == "random_projection":
        target_dim = int(params.get("target_dim", 128))
        seed = int(params.get("seed", 42))
        rng = np.random.default_rng(seed)
        matrix = rng.normal(0.0, 1.0 / np.sqrt(target_dim), size=(target_dim, vector.shape[0])).astype(np.float32)
        return _l2_normalize(matrix @ vector)

    if method == "quantization":
        levels = max(int(params.get("levels", 32)), 2)
        clipped = np.clip(vector, -1.0, 1.0)
        scaled = (clipped + 1.0) / 2.0
        quantized = np.round(scaled * (levels - 1)) / (levels - 1)
        quantized = quantized * 2.0 - 1.0
        return _l2_normalize(quantized.astype(np.float32))

    if method == "cancellable_transform":
        seed = int(params.get("seed", 42))
        mix_ratio = float(params.get("mix_ratio", 1.0))
        mix_ratio = max(0.0, min(1.0, mix_ratio))
        rng = np.random.default_rng(seed)
        permutation = rng.permutation(vector.shape[0])
        signs = rng.choice(np.array([-1.0, 1.0], dtype=np.float32), size=vector.shape[0])
        keyed = vector[permutation] * signs
        transformed = (1.0 - mix_ratio) * vector + mix_ratio * keyed
        return _l2_normalize(transformed.astype(np.float32))

    raise ValueError(f"Unsupported embedding method: {method}")


def _l2_normalize(vector: np.ndarray) -> np.ndarray:
    return (vector / (np.linalg.norm(vector) + 1e-12)).astype(np.float32)


def _odd_kernel(kernel_size: int) -> int:
    kernel_size = max(kernel_size, 1)
    return kernel_size if kernel_size % 2 == 1 else kernel_size + 1
