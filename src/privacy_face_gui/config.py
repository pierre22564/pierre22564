from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppConfig:
    project_root: Path
    min_faces_per_person: int = 20
    test_size: float = 0.3
    random_state: int = 42
    embedding_dim: int = 512
    image_size: int = 112

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"

    @property
    def artifacts_dir(self) -> Path:
        return self.project_root / "artifacts"

    @property
    def results_dir(self) -> Path:
        return self.project_root / "results"

    @property
    def experiments_dir(self) -> Path:
        return self.project_root / "experiments"

    @property
    def lfw_root(self) -> Path:
        return self.data_dir / "lfw_funneled"

    @property
    def arcface_model_path(self) -> Path:
        return self.models_dir / "w600k_r50.onnx"

    @property
    def embeddings_cache_path(self) -> Path:
        return self.artifacts_dir / f"embeddings_lfw_min{self.min_faces_per_person}.npz"

    @property
    def classifier_cache_path(self) -> Path:
        return self.artifacts_dir / f"knn_lfw_min{self.min_faces_per_person}.joblib"

    @property
    def dataset_summary_path(self) -> Path:
        return self.artifacts_dir / f"dataset_summary_min{self.min_faces_per_person}.joblib"

    @classmethod
    def discover(cls, start: Path | None = None) -> "AppConfig":
        current = (start or Path.cwd()).resolve()
        candidates = [current, *current.parents]
        for candidate in candidates:
            if (candidate / "data").exists() and (candidate / "models").exists():
                config = cls(project_root=candidate)
                config.ensure_directories()
                return config
        config = cls(project_root=current)
        config.ensure_directories()
        return config

    def ensure_directories(self) -> None:
        for path in [
            self.data_dir,
            self.models_dir,
            self.artifacts_dir,
            self.results_dir,
            self.experiments_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)
