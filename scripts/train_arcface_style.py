from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from privacy_face_gui.pipeline import PrivacyFacePipeline  # noqa: E402


@dataclass
class ModelComparison:
    model_name: str
    top1_accuracy: float
    rejection_rate: float
    overall_success_rate: float
    mean_distance: float
    threshold: float
    num_samples: int
    embedding_dim: int


class EmbeddingBackbone(nn.Module):
    def __init__(self, in_dim: int = 512, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=1)


class ArcMarginProduct(nn.Module):
    def __init__(self, in_features: int, out_features: int, s: float = 30.0, m: float = 0.35):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        sine = torch.sqrt(torch.clamp(1.0 - cosine**2, min=1e-9))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        logits = one_hot * phi + (1.0 - one_hot) * cosine
        return logits * self.s


def evaluate_embeddings(
    embeddings: np.ndarray,
    labels: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    known_q: float = 0.95,
) -> ModelComparison:
    train_embeddings = embeddings[train_idx]
    train_labels = labels[train_idx]
    knn = KNeighborsClassifier(n_neighbors=1, metric="cosine")
    knn.fit(train_embeddings, train_labels)

    known_distances: list[float] = []
    for idx in test_idx:
        distance, _ = knn.kneighbors(embeddings[idx][None, :], n_neighbors=1, return_distance=True)
        known_distances.append(float(distance[0][0]))
    threshold = float(np.quantile(np.asarray(known_distances), known_q))

    predictions: list[int] = []
    distances: list[float] = []
    rejected = 0
    for idx in test_idx:
        distance, neighbor_idx = knn.kneighbors(embeddings[idx][None, :], n_neighbors=1, return_distance=True)
        pred_distance = float(distance[0][0])
        pred_label = int(train_labels[int(neighbor_idx[0][0])])
        distances.append(pred_distance)
        if pred_distance > threshold:
            rejected += 1
            predictions.append(-1)
        else:
            predictions.append(pred_label)

    truths = labels[test_idx]
    accepted_mask = np.array(predictions) != -1
    top1_accuracy = float(accuracy_score(truths[accepted_mask], np.array(predictions)[accepted_mask])) if accepted_mask.any() else 0.0
    rejection_rate = float(rejected / len(test_idx))
    overall_success_rate = float(top1_accuracy * (1.0 - rejection_rate))
    mean_distance = float(np.mean(distances))

    return ModelComparison(
        model_name="",
        top1_accuracy=top1_accuracy,
        rejection_rate=rejection_rate,
        overall_success_rate=overall_success_rate,
        mean_distance=mean_distance,
        threshold=threshold,
        num_samples=int(len(test_idx)),
        embedding_dim=int(embeddings.shape[1]),
    )


def train_model(pipeline: PrivacyFacePipeline, epochs: int = 40, batch_size: int = 128, learning_rate: float = 1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x = torch.from_numpy(pipeline.embeddings[pipeline.train_idx]).float()
    train_y = torch.from_numpy(pipeline.dataset.labels[pipeline.train_idx]).long()
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)

    model = EmbeddingBackbone(in_dim=train_x.shape[1], hidden_dim=256, out_dim=128).to(device)
    head = ArcMarginProduct(in_features=128, out_features=len(pipeline.dataset.label_to_id)).to(device)
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=learning_rate, weight_decay=1e-4)

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        model.train()
        head.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0
        for embeddings, labels in loader:
            embeddings = embeddings.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            adapted = model(embeddings)
            logits = head(adapted, labels)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = logits.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        epoch_loss = running_loss / max(running_total, 1)
        epoch_acc = running_correct / max(running_total, 1)
        history.append({"epoch": epoch, "train_loss": epoch_loss, "train_accuracy": epoch_acc})
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs} - loss={epoch_loss:.4f} - train_acc={epoch_acc:.4f}")
    return model, history


def adapt_embeddings(model: nn.Module, embeddings: np.ndarray, batch_size: int = 256) -> np.ndarray:
    device = next(model.parameters()).device
    loader = DataLoader(torch.from_numpy(embeddings).float(), batch_size=batch_size, shuffle=False)
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0).astype(np.float32)


def save_training_figure(history_df: pd.DataFrame, output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history_df["epoch"], history_df["train_loss"], marker="o")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.25)
    axes[1].plot(history_df["epoch"], history_df["train_accuracy"], marker="o")
    axes[1].set_title("Training accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_ylim(0.0, 1.02)
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def save_comparison_figure(comparison_df: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    metrics = ["top1_accuracy", "overall_success_rate", "rejection_rate"]
    x = np.arange(len(metrics))
    width = 0.35
    axis.bar(x - width / 2, comparison_df.iloc[0][metrics].to_numpy(dtype=float), width=width, label=comparison_df.iloc[0]["model_name"])
    axis.bar(x + width / 2, comparison_df.iloc[1][metrics].to_numpy(dtype=float), width=width, label=comparison_df.iloc[1]["model_name"])
    axis.set_xticks(x)
    axis.set_xticklabels(metrics)
    axis.set_ylim(0.0, 1.02)
    axis.set_title("Frozen pre-trained ArcFace vs dataset-adapted ArcFace-style head")
    axis.set_ylabel("score")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    pipeline = PrivacyFacePipeline.discover(PROJECT_ROOT)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pretrained_baseline = evaluate_embeddings(
        embeddings=pipeline.embeddings,
        labels=pipeline.dataset.labels,
        train_idx=pipeline.train_idx,
        test_idx=pipeline.test_idx,
    )
    pretrained_baseline.model_name = "Pre-trained ArcFace"
    print(
        f"Pre-trained ArcFace - top1_accuracy={pretrained_baseline.top1_accuracy:.4f} "
        f"overall_success_rate={pretrained_baseline.overall_success_rate:.4f}"
    )

    model, history = train_model(pipeline=pipeline)
    adapted_embeddings = adapt_embeddings(model, pipeline.embeddings)
    trained_result = evaluate_embeddings(
        embeddings=adapted_embeddings,
        labels=pipeline.dataset.labels,
        train_idx=pipeline.train_idx,
        test_idx=pipeline.test_idx,
    )
    trained_result.model_name = "Dataset-adapted ArcFace head"
    print(
        f"Dataset-adapted ArcFace head - top1_accuracy={trained_result.top1_accuracy:.4f} "
        f"overall_success_rate={trained_result.overall_success_rate:.4f}"
    )

    artifacts_dir = pipeline.config.artifacts_dir
    results_dir = pipeline.config.results_dir
    docs_data_dir = PROJECT_ROOT / "docs" / "data"
    docs_assets_dir = PROJECT_ROOT / "docs" / "assets"
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    docs_assets_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "arcface_style_embedding_head.pt"
    torch.save({"state_dict": model.state_dict(), "embedding_dim": trained_result.embedding_dim}, model_path)
    np.savez_compressed(
        artifacts_dir / "arcface_style_adapted_embeddings_lfw_min20.npz",
        embeddings=adapted_embeddings,
        labels=pipeline.dataset.labels,
        image_paths=pipeline.dataset.frame["image_path"].to_numpy(),
    )
    joblib.dump(
        {"history": history, "comparison": [asdict(pretrained_baseline), asdict(trained_result)]},
        artifacts_dir / "arcface_style_training_summary.joblib",
    )

    history_df = pd.DataFrame(history)
    comparison_df = pd.DataFrame([asdict(pretrained_baseline), asdict(trained_result)])
    history_df.to_csv(results_dir / "arcface_style_training_history.csv", index=False)
    comparison_df.to_csv(results_dir / "arcface_training_comparison.csv", index=False)

    save_training_figure(history_df, docs_assets_dir / "arcface_style_training_curve.png")
    save_comparison_figure(comparison_df, docs_assets_dir / "arcface_training_comparison.png")

    payload = {
        "training_protocol": {
            "dataset": "LFW filtered with min_faces_per_person=20",
            "num_identities": int(pipeline.dataset.filtered_identities),
            "num_images": int(pipeline.dataset.filtered_images),
            "train_samples": int(len(pipeline.train_idx)),
            "test_samples": int(len(pipeline.test_idx)),
            "epochs": 40,
            "batch_size": 128,
            "embedding_dim": 128,
            "device": str(device),
            "note": "This is a lightweight dataset-specific ArcFace-style adaptation trained on top of frozen pre-trained ArcFace embeddings, not a full retraining of the original ArcFace backbone.",
        },
        "comparison": [asdict(pretrained_baseline), asdict(trained_result)],
    }
    (docs_data_dir / "arcface_training_comparison.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved comparison JSON to {docs_data_dir / 'arcface_training_comparison.json'}")


if __name__ == "__main__":
    main()
