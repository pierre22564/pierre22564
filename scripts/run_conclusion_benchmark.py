from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from privacy_face_gui.pipeline import PrivacyFacePipeline  # noqa: E402


CONCLUSION_SWEEP_CONFIGS = {
    "gaussian_blur": {"parameter_name": "kernel_size", "values": [5, 11, 21, 31, 41, 61, 81], "base_params": {}},
    "embedding_noise": {"parameter_name": "sigma", "values": [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3], "base_params": {}},
    "embedding_dp_laplace": {
        "parameter_name": "epsilon",
        "values": [32.0, 16.0, 8.0, 4.0, 2.0, 1.0, 0.5],
        "base_params": {"sensitivity": 1.0},
    },
    "random_projection": {"parameter_name": "target_dim", "values": [512, 256, 128, 64, 32, 16], "base_params": {"seed": 42}},
    "quantization": {"parameter_name": "levels", "values": [256, 128, 64, 32, 16, 8, 4], "base_params": {}},
    "cancellable_transform": {"parameter_name": "mix_ratio", "values": [0.2, 0.4, 0.6, 0.8, 1.0], "base_params": {"seed": 42}},
}

METHOD_LABELS = {
    "gaussian_blur": "Gaussian Blur",
    "embedding_noise": "Noise Injection",
    "embedding_dp_laplace": "Differential Privacy",
    "random_projection": "Random Projection",
    "quantization": "Quantization",
    "cancellable_transform": "Cancellable Transformation",
}

METHOD_GROUPS = {
    "gaussian_blur": "Before ArcFace (photo-space)",
    "embedding_noise": "After ArcFace (embedding-space)",
    "embedding_dp_laplace": "After ArcFace (embedding-space)",
    "random_projection": "After ArcFace (embedding-space)",
    "quantization": "After ArcFace (embedding-space)",
    "cancellable_transform": "After ArcFace (embedding-space)",
}


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def format_parameter_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    if float(value).is_integer():
        return str(int(value))
    return f"{float(value):.3f}"


def add_overall_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["overall_success_rate"] = df["top1_accuracy"] * (1.0 - df["rejection_rate"])
    return df


def compute_operating_point_summary(benchmark_df: pd.DataFrame, target_utility: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method, method_df in benchmark_df.groupby("method"):
        eligible = method_df[method_df["overall_success_rate"] >= target_utility].copy()
        feasible = not eligible.empty
        source_df = eligible if feasible else method_df.copy()
        source_df = source_df.sort_values(
            by=["unknown_rejection_rate", "overall_success_rate"],
            ascending=[False, False],
        )
        best = source_df.iloc[0]
        rows.append(
            {
                "method": method_label(method),
                "raw_method": method,
                "parameter_name": str(best["parameter_name"]),
                "parameter_value": best["parameter_value"],
                "parameter_display": f"{best['parameter_name']}={format_parameter_value(best['parameter_value'])}",
                "overall_success_rate": float(best["overall_success_rate"]),
                "unknown_rejection_rate": float(best["unknown_rejection_rate"]),
                "top1_accuracy": float(best["top1_accuracy"]),
                "rejection_rate": float(best["rejection_rate"]),
                "feasible_at_target_utility": feasible,
                "has_formal_privacy_guarantee": method == "embedding_dp_laplace",
            }
        )
    summary_df = pd.DataFrame(rows)
    if summary_df.empty:
        return summary_df
    return summary_df.sort_values(
        by=["feasible_at_target_utility", "unknown_rejection_rate", "overall_success_rate"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_conclusion_text(summary_df: pd.DataFrame, target_utility: float) -> str:
    feasible = summary_df[summary_df["feasible_at_target_utility"] == True]  # noqa: E712
    dp_rows = summary_df[summary_df["raw_method"] == "embedding_dp_laplace"]
    if feasible.empty:
        best = summary_df.iloc[0]
        text = (
            f"No method reached the target utility level of {target_utility:.2f}. "
            f"The closest operating point was {best['method']} with {best['parameter_display']}, "
            f"overall success rate {best['overall_success_rate']:.3f}, and unknown rejection rate {best['unknown_rejection_rate']:.3f}."
        )
        if not dp_rows.empty:
            dp_row = dp_rows.iloc[0]
            text += (
                f" Differential Privacy remained below the utility target in this benchmark. "
                f"Its strongest tested point was epsilon={format_parameter_value(dp_row['parameter_value'])}, "
                f"with overall success rate {dp_row['overall_success_rate']:.3f} and unknown rejection rate {dp_row['unknown_rejection_rate']:.3f}."
            )
        return text

    best = feasible.iloc[0]
    text = (
        f"For the chosen utility target ({target_utility:.2f} overall success rate), among the methods that reached this utility level, "
        f"the best empirical privacy proxy is "
        f"{best['method']} with {best['parameter_display']}. "
        f"At this operating point, the overall success rate is {best['overall_success_rate']:.3f} "
        f"and the unknown rejection rate is {best['unknown_rejection_rate']:.3f}."
    )
    if not dp_rows.empty:
        dp_row = dp_rows.iloc[0]
        if bool(dp_row["feasible_at_target_utility"]):
            text += (
                f" Differential Privacy is the only method here with a formal privacy parameter (epsilon). "
                f"Its best feasible point is epsilon={format_parameter_value(dp_row['parameter_value'])}, "
                f"with overall success rate {dp_row['overall_success_rate']:.3f} and unknown rejection rate {dp_row['unknown_rejection_rate']:.3f}."
            )
        else:
            text += (
                f" Differential Privacy is the only method here with a formal privacy parameter (epsilon), "
                f"but in this benchmark it did not reach the target utility level. "
                f"Its best tested point was epsilon={format_parameter_value(dp_row['parameter_value'])}, "
                f"with overall success rate {dp_row['overall_success_rate']:.3f} and unknown rejection rate {dp_row['unknown_rejection_rate']:.3f}."
            )
    return text


def make_tradeoff_scatter_figure(benchmark_df: pd.DataFrame, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5))
    for method, method_df in benchmark_df.groupby("method"):
        axis.scatter(
            method_df["overall_success_rate"],
            method_df["unknown_rejection_rate"],
            label=method_label(method),
            s=70,
            alpha=0.85,
        )
    axis.set_xlabel("Overall success rate (utility)")
    axis.set_ylabel("Unknown rejection rate (privacy proxy)")
    axis.set_title("Utility vs privacy trade-off")
    axis.set_xlim(0.0, 1.02)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend(loc="best", fontsize=8)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def make_best_method_bar_figure(summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary_df.copy().sort_values("unknown_rejection_rate", ascending=False)
    figure, axis = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(plot_df))
    width = 0.38
    axis.bar(x - width / 2, plot_df["overall_success_rate"], width=width, label="overall success")
    axis.bar(x + width / 2, plot_df["unknown_rejection_rate"], width=width, label="unknown rejection")
    axis.set_xticks(x)
    axis.set_xticklabels(plot_df["method"], rotation=20, ha="right")
    axis.set_ylim(0.0, 1.02)
    axis.set_ylabel("score")
    axis.set_title("Best operating point per method")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(loc="best")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main() -> None:
    pipeline = PrivacyFacePipeline.discover(PROJECT_ROOT)
    threshold = float(pipeline.threshold)
    known_limit = 20
    unknown_count = 20
    target_utility = 0.80

    rows: list[dict[str, object]] = []
    for method, config in CONCLUSION_SWEEP_CONFIGS.items():
        parameter_name = config["parameter_name"]
        base_params = dict(config.get("base_params", {}))
        print(f"\n[{method_label(method)}]")
        for parameter_value in config["values"]:
            params = dict(base_params)
            params[parameter_name] = parameter_value
            print(f"  - {parameter_name}={format_parameter_value(parameter_value)}")
            known_result = pipeline.evaluate_method(method=method, params=params, threshold=threshold, limit=known_limit)
            unknown_df = pipeline.evaluate_unknown_images(
                image_paths=pipeline.sample_unknown_paths(count=unknown_count),
                method=method,
                params=params,
                threshold=threshold,
            )
            unknown_rejection_rate = float(unknown_df["rejected"].mean()) if not unknown_df.empty else 0.0
            rows.append(
                {
                    "method": method,
                    "parameter_name": parameter_name,
                    "parameter_value": parameter_value,
                    "top1_accuracy": known_result.top1_accuracy,
                    "rejection_rate": known_result.rejection_rate,
                    "mean_distance": known_result.mean_distance,
                    "unknown_rejection_rate": unknown_rejection_rate,
                    "threshold": threshold,
                    "num_samples": known_result.num_samples,
                }
            )

    benchmark_df = add_overall_success_rate(pd.DataFrame(rows))
    summary_df = compute_operating_point_summary(benchmark_df, target_utility)
    conclusion_text = build_conclusion_text(summary_df, target_utility)

    results_dir = pipeline.config.results_dir
    docs_data_dir = PROJECT_ROOT / "docs" / "data"
    docs_assets_dir = PROJECT_ROOT / "docs" / "assets"
    results_dir.mkdir(parents=True, exist_ok=True)
    docs_data_dir.mkdir(parents=True, exist_ok=True)
    docs_assets_dir.mkdir(parents=True, exist_ok=True)
    benchmark_df.to_csv(results_dir / "conclusion_benchmark_latest.csv", index=False)
    summary_df.to_csv(results_dir / "conclusion_summary_latest.csv", index=False)

    payload = {
        "type": "conclusion_benchmark",
        "threshold": threshold,
        "known_limit": known_limit,
        "unknown_count": unknown_count,
        "target_utility": target_utility,
        "rows": benchmark_df.to_dict(orient="records"),
        "summary_rows": summary_df.to_dict(orient="records"),
        "conclusion_text": conclusion_text,
    }
    saved_path = pipeline.save_experiment("conclusion_benchmark", payload)
    (docs_data_dir / "conclusion_benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    method_table = pd.DataFrame(
        [
            {
                "method": method_label(method),
                "applied_to": METHOD_GROUPS[method],
                "parameter_name": config["parameter_name"],
                "tested_values": ", ".join(format_parameter_value(value) for value in config["values"]),
            }
            for method, config in CONCLUSION_SWEEP_CONFIGS.items()
        ]
    )
    method_table.to_csv(docs_data_dir / "method_parameter_summary.csv", index=False)
    make_tradeoff_scatter_figure(benchmark_df, docs_assets_dir / "conclusion_tradeoff_scatter.png")
    make_best_method_bar_figure(summary_df, docs_assets_dir / "conclusion_best_methods.png")

    print("\nSaved benchmark to:")
    print(saved_path)
    print("\nConclusion:")
    print(conclusion_text)


if __name__ == "__main__":
    main()
