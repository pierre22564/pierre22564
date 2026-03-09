from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .config import AppConfig


@dataclass
class ArcFaceEmbedder:
    session: ort.InferenceSession
    input_name: str
    output_name: str
    image_size: int

    @classmethod
    def from_config(cls, config: AppConfig) -> "ArcFaceEmbedder":
        model_path = config.arcface_model_path
        if not model_path.exists():
            raise FileNotFoundError(
                f"ArcFace model not found at {model_path}. Place w600k_r50.onnx in the models directory."
            )
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        return cls(
            session=session,
            input_name=session.get_inputs()[0].name,
            output_name=session.get_outputs()[0].name,
            image_size=config.image_size,
        )

    def preprocess(self, image_rgb: np.ndarray) -> np.ndarray:
        image = cv2.resize(image_rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR).astype(np.float32)
        image = (image - 127.5) / 127.5
        return np.transpose(image, (2, 0, 1))[None, ...]

    def embed(self, image_rgb: np.ndarray) -> np.ndarray:
        embedding = self.session.run([self.output_name], {self.input_name: self.preprocess(image_rgb)})[0][0]
        embedding = embedding / (np.linalg.norm(embedding) + 1e-12)
        return embedding.astype(np.float32)
