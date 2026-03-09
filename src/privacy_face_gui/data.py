from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .config import AppConfig


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


@dataclass(frozen=True)
class DatasetIndex:
    frame: pd.DataFrame
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    people_counts: Counter

    @property
    def image_paths(self) -> list[Path]:
        return [Path(path) for path in self.frame["image_path"].tolist()]

    @property
    def labels(self) -> np.ndarray:
        return self.frame["label_id"].to_numpy(dtype=np.int32)

    @property
    def selected_people(self) -> list[str]:
        return sorted(self.label_to_id.keys())

    @property
    def total_identities(self) -> int:
        return len(self.people_counts)

    @property
    def total_images(self) -> int:
        return int(sum(self.people_counts.values()))

    @property
    def filtered_identities(self) -> int:
        return int(self.frame["label_name"].nunique())

    @property
    def filtered_images(self) -> int:
        return len(self.frame)


def build_dataset_index(config: AppConfig) -> DatasetIndex:
    if not config.lfw_root.exists():
        raise FileNotFoundError(
            f"LFW dataset not found at {config.lfw_root}. Run the notebook or download the dataset first."
        )

    people_counts: Counter = Counter()
    records: list[dict[str, object]] = []

    person_dirs = sorted(path for path in config.lfw_root.iterdir() if path.is_dir())
    for person_dir in person_dirs:
        count = len([path for path in person_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS])
        people_counts[person_dir.name] = count

    selected_people = [name for name, count in people_counts.items() if count >= config.min_faces_per_person]
    label_to_id = {name: idx for idx, name in enumerate(sorted(selected_people))}
    id_to_label = {idx: name for name, idx in label_to_id.items()}

    for label_name in selected_people:
        for image_path in sorted((config.lfw_root / label_name).iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            records.append(
                {
                    "image_path": str(image_path.resolve()),
                    "label_name": label_name,
                    "label_id": label_to_id[label_name],
                }
            )

    frame = pd.DataFrame.from_records(records).sort_values(["label_name", "image_path"]).reset_index(drop=True)
    return DatasetIndex(
        frame=frame,
        label_to_id=label_to_id,
        id_to_label=id_to_label,
        people_counts=people_counts,
    )


def read_rgb_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_bgr_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image
