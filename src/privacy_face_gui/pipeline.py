from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from .anonymization import EMBEDDING_METHODS, IMAGE_METHODS, apply_embedding_method, apply_image_method
from .arcface import ArcFaceEmbedder
from .config import AppConfig
from .data import DatasetIndex, build_dataset_index, read_rgb_image


@dataclass
class QueryResult:
    method: str
    query_embedding: np.ndarray
    query_image: np.ndarray
    protected_image: np.ndarray | None
    predicted_label: str
    predicted_distance: float
    matched_image_path: Path
    matched_label: str
    is_known: bool
    is_correct: bool | None


@dataclass
class EvaluationResult:
    method: str
    params: dict[str, Any]
    top1_accuracy: float
    rejection_rate: float
    mean_distance: float
    num_samples: int
    num_rejected: int
    threshold: float


class PrivacyFacePipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        self.dataset = build_dataset_index(config)
        self.dataset_signature = self._build_dataset_signature()
        self.embedder = ArcFaceEmbedder.from_config(config)
        self.embeddings = self._load_or_compute_embeddings()
        self.train_idx, self.test_idx, self.knn = self._load_or_train_classifier()
        self.threshold = self._estimate_unknown_threshold()

    @classmethod
    def discover(cls, start: Path | None = None) -> "PrivacyFacePipeline":
        return cls(AppConfig.discover(start))

    def _load_or_compute_embeddings(self) -> np.ndarray:
        cache_path = self.config.embeddings_cache_path
        if cache_path.exists():
            cached = np.load(cache_path, allow_pickle=True)
            standardized = self._standardize_cached_embeddings(cached)
            if standardized is not None:
                return standardized

        embeddings = np.zeros((len(self.dataset.frame), self.config.embedding_dim), dtype=np.float32)
        for idx, image_path in enumerate(self.dataset.frame["image_path"].tolist()):
            embeddings[idx] = self.embedder.embed(read_rgb_image(image_path))

        np.savez_compressed(
            cache_path,
            embeddings=embeddings,
            y=self.dataset.labels,
            image_paths=self.dataset.frame["image_path"].to_numpy(),
            min_faces=np.array([self.config.min_faces_per_person], dtype=np.int32),
            dataset_signature=np.array([self.dataset_signature]),
        )
        return embeddings

    def _load_or_train_classifier(self) -> tuple[np.ndarray, np.ndarray, KNeighborsClassifier]:
        train_idx, test_idx = train_test_split(
            np.arange(len(self.dataset.frame)),
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=self.dataset.labels,
        )
        cache_path = self.config.classifier_cache_path
        if cache_path.exists():
            payload = joblib.load(cache_path)
            if (
                payload.get("embed_cache") == str(self.config.embeddings_cache_path)
                and payload.get("n_samples") == len(self.dataset.frame)
                and payload.get("dataset_signature") == self.dataset_signature
            ):
                return payload["train_idx"], payload["test_idx"], payload["knn"]

        knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
        knn.fit(self.embeddings[train_idx], self.dataset.labels[train_idx])
        joblib.dump(
            {
                "knn": knn,
                "train_idx": train_idx,
                "test_idx": test_idx,
                "embed_cache": str(self.config.embeddings_cache_path),
                "n_samples": len(self.dataset.frame),
                "dataset_signature": self.dataset_signature,
            },
            cache_path,
        )
        return train_idx, test_idx, knn

    def _build_dataset_signature(self) -> str:
        joined = "\n".join(self.dataset.frame["image_path"].tolist())
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()

    def _standardize_cached_embeddings(self, cached: np.lib.npyio.NpzFile) -> np.ndarray | None:
        if "embeddings" not in cached.files:
            return None

        embeddings = cached["embeddings"].astype(np.float32)
        if len(embeddings) != len(self.dataset.frame):
            return None

        cached_signature = None
        if "dataset_signature" in cached.files:
            cached_signature = str(cached["dataset_signature"][0])
        if cached_signature == self.dataset_signature:
            return embeddings

        if "image_paths" not in cached.files:
            return None

        cached_paths = [str(path) for path in cached["image_paths"].tolist()]
        current_paths = self.dataset.frame["image_path"].tolist()
        if set(cached_paths) != set(current_paths):
            return None

        path_to_position = {path: idx for idx, path in enumerate(cached_paths)}
        reordered = np.asarray([embeddings[path_to_position[path]] for path in current_paths], dtype=np.float32)
        np.savez_compressed(
            self.config.embeddings_cache_path,
            embeddings=reordered,
            y=self.dataset.labels,
            image_paths=np.array(current_paths),
            min_faces=np.array([self.config.min_faces_per_person], dtype=np.int32),
            dataset_signature=np.array([self.dataset_signature]),
        )
        return reordered

    def _estimate_unknown_threshold(self, known_q: float = 0.95, limit: int = 300) -> float:
        gallery = self.embeddings[self.train_idx]
        known_distances: list[float] = []
        for idx in self.test_idx[:limit]:
            query = self.embeddings[idx]
            _, distance = _nearest_neighbor(query, gallery, self.dataset.labels[self.train_idx])
            known_distances.append(distance)
        threshold = float(np.quantile(known_distances, known_q))
        return max(0.0, min(1.0, threshold))

    def dataset_summary(self) -> dict[str, Any]:
        return {
            "total_identities": self.dataset.total_identities,
            "total_images": self.dataset.total_images,
            "filtered_identities": self.dataset.filtered_identities,
            "filtered_images": self.dataset.filtered_images,
            "min_faces_per_person": self.config.min_faces_per_person,
            "threshold": self.threshold,
        }

    def get_sample_paths(self, count: int = 6, per_identity: int = 2, seed: int = 42) -> list[list[Path]]:
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(self.dataset.selected_people, size=min(count, len(self.dataset.selected_people)), replace=False))
        grid: list[list[Path]] = []
        for label_name in selected:
            frame = self.dataset.frame[self.dataset.frame["label_name"] == label_name].head(per_identity)
            grid.append([Path(path) for path in frame["image_path"].tolist()])
        return grid

    def evaluate_method(
        self,
        method: str,
        params: dict[str, Any] | None = None,
        threshold: float | None = None,
        limit: int | None = None,
    ) -> EvaluationResult:
        params = params or {}
        threshold = self.threshold if threshold is None else threshold
        test_idx = self.test_idx[:limit] if limit else self.test_idx
        train_embeddings = self._prepare_gallery(method, params)

        predictions: list[int] = []
        distances: list[float] = []
        rejected = 0

        for idx in test_idx:
            query_embedding = self._prepare_query_embedding(idx, method, params)
            pred_label, pred_distance = _nearest_neighbor(query_embedding, train_embeddings, self.dataset.labels[self.train_idx])
            distances.append(pred_distance)
            if pred_distance > threshold:
                rejected += 1
                predictions.append(-1)
            else:
                predictions.append(pred_label)

        truths = self.dataset.labels[test_idx]
        accepted_mask = np.array(predictions) != -1
        if accepted_mask.any():
            acc = accuracy_score(truths[accepted_mask], np.array(predictions)[accepted_mask])
        else:
            acc = 0.0
        return EvaluationResult(
            method=method,
            params=params,
            top1_accuracy=float(acc),
            rejection_rate=float(rejected / len(test_idx)),
            mean_distance=float(np.mean(distances)),
            num_samples=len(test_idx),
            num_rejected=rejected,
            threshold=float(threshold),
        )

    def evaluate_unknown_images(
        self,
        image_paths: list[Path],
        method: str,
        params: dict[str, Any] | None = None,
        threshold: float | None = None,
    ) -> pd.DataFrame:
        params = params or {}
        threshold = self.threshold if threshold is None else threshold
        gallery = self._prepare_gallery(method, params)
        rows: list[dict[str, Any]] = []

        for image_path in image_paths:
            image_rgb = read_rgb_image(image_path)
            query_embedding, protected_image = self._build_query_embedding_from_image(image_rgb, method, params)
            pred_label, pred_distance = _nearest_neighbor(query_embedding, gallery, self.dataset.labels[self.train_idx])
            is_rejected = pred_distance > threshold
            rows.append(
                {
                    "image_path": str(image_path),
                    "predicted_label": self.dataset.id_to_label[int(pred_label)],
                    "distance": float(pred_distance),
                    "rejected": bool(is_rejected),
                    "method": method,
                }
            )
            _ = protected_image
        return pd.DataFrame(rows)

    def run_parameter_sweep(
        self,
        method: str,
        parameter_name: str,
        parameter_values: list[float | int],
        base_params: dict[str, Any] | None = None,
        threshold: float | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        base_params = base_params or {}
        rows: list[dict[str, Any]] = []
        for value in parameter_values:
            params = dict(base_params)
            params[parameter_name] = value
            result = self.evaluate_method(method=method, params=params, threshold=threshold, limit=limit)
            row = {
                "method": method,
                "parameter_name": parameter_name,
                "parameter_value": value,
                "top1_accuracy": result.top1_accuracy,
                "rejection_rate": result.rejection_rate,
                "mean_distance": result.mean_distance,
                "threshold": result.threshold,
                "num_samples": result.num_samples,
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def query_image(
        self,
        image_rgb: np.ndarray,
        method: str,
        params: dict[str, Any] | None = None,
        known_label: str | None = None,
        threshold: float | None = None,
    ) -> QueryResult:
        params = params or {}
        threshold = self.threshold if threshold is None else threshold
        query_embedding, protected_image = self._build_query_embedding_from_image(image_rgb, method, params)
        gallery = self._prepare_gallery(method, params)
        pred_label_id, pred_distance, matched_position = _nearest_neighbor(
            query_embedding,
            gallery,
            self.dataset.labels[self.train_idx],
            return_position=True,
        )

        matched_global_idx = int(self.train_idx[matched_position])
        matched_label = self.dataset.id_to_label[int(pred_label_id)]
        is_known = pred_distance <= threshold
        is_correct = None if known_label is None or not is_known else matched_label == known_label
        return QueryResult(
            method=method,
            query_embedding=query_embedding,
            query_image=image_rgb,
            protected_image=protected_image,
            predicted_label=matched_label if is_known else "UNKNOWN",
            predicted_distance=float(pred_distance),
            matched_image_path=Path(self.dataset.frame.iloc[matched_global_idx]["image_path"]),
            matched_label=matched_label,
            is_known=is_known,
            is_correct=is_correct,
        )

    def sample_unknown_paths(self, count: int = 12, seed: int = 42) -> list[Path]:
        excluded_people = [
            name for name, count_images in self.dataset.people_counts.items() if count_images < self.config.min_faces_per_person
        ]
        paths: list[Path] = []
        for name in excluded_people:
            for image_path in sorted((self.config.lfw_root / name).iterdir()):
                if image_path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    paths.append(image_path)
        rng = np.random.default_rng(seed)
        if not paths:
            return []
        selected = rng.choice(np.array(paths, dtype=object), size=min(count, len(paths)), replace=False)
        return [Path(path) for path in selected.tolist()]

    def save_experiment(self, name: str, payload: dict[str, Any]) -> Path:
        slug = _slugify(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = self.config.experiments_dir / f"{timestamp}_{slug}.json"
        serializable = _to_jsonable(payload)
        path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
        return path

    def load_saved_experiments(self) -> list[Path]:
        return sorted(self.config.experiments_dir.glob("*.json"), reverse=True)

    def _prepare_gallery(self, method: str, params: dict[str, Any]) -> np.ndarray:
        base_gallery = self.embeddings[self.train_idx]
        if method in IMAGE_METHODS:
            return base_gallery
        if method in EMBEDDING_METHODS:
            transformed = [apply_embedding_method(embedding, method, params) for embedding in base_gallery]
            return np.asarray(transformed, dtype=np.float32)
        raise ValueError(f"Unsupported method: {method}")

    def _prepare_query_embedding(self, idx: int, method: str, params: dict[str, Any]) -> np.ndarray:
        image_rgb = read_rgb_image(self.dataset.frame.iloc[idx]["image_path"])
        query_embedding, _ = self._build_query_embedding_from_image(image_rgb, method, params, use_cached_embedding=True, idx=idx)
        return query_embedding

    def _build_query_embedding_from_image(
        self,
        image_rgb: np.ndarray,
        method: str,
        params: dict[str, Any],
        use_cached_embedding: bool = False,
        idx: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        if method in IMAGE_METHODS:
            protected_image = apply_image_method(image_rgb, method, params)
            return self.embedder.embed(protected_image), protected_image

        if method in EMBEDDING_METHODS:
            if use_cached_embedding and idx is not None:
                base_embedding = self.embeddings[idx]
            else:
                base_embedding = self.embedder.embed(image_rgb)
            return apply_embedding_method(base_embedding, method, params), None

        raise ValueError(f"Unsupported method: {method}")


def _nearest_neighbor(
    query_embedding: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_labels: np.ndarray,
    return_position: bool = False,
) -> tuple[int, float] | tuple[int, float, int]:
    distances = 1.0 - np.clip(gallery_embeddings @ query_embedding, -1.0, 1.0)
    position = int(np.argmin(distances))
    label = int(gallery_labels[position])
    distance = float(distances[position])
    if return_position:
        return label, distance, position
    return label, distance


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(1.0 - np.clip(np.dot(a, b), -1.0, 1.0))


def _slugify(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized.strip("-") or "experiment"


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_jsonable(item) for item in value]
    return value
