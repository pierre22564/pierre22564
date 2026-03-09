from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from privacy_face_gui.anonymization import METHOD_SPECS, build_visual_embedding_map  # noqa: E402
from privacy_face_gui.method_notes import METHOD_NOTES  # noqa: E402
from privacy_face_gui.pipeline import PrivacyFacePipeline, QueryResult  # noqa: E402


st.set_page_config(
    page_title="Privacy-Preserving Face Recognition",
    page_icon="FR",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      :root {
        --bg: #f2efe7;
        --card: #fffaf2;
        --ink: #172121;
        --muted: #5b635c;
        --accent: #c75c2f;
        --accent-2: #2d6a6a;
        --line: #d7cfc2;
      }
      .stApp {
        background:
          radial-gradient(circle at top left, rgba(199, 92, 47, 0.15), transparent 25%),
          radial-gradient(circle at top right, rgba(45, 106, 106, 0.15), transparent 24%),
          linear-gradient(180deg, #f7f2e8 0%, #efe8db 100%);
        color: var(--ink);
      }
      .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1400px;
      }
      .hero {
        padding: 1.2rem 1.4rem;
        border: 1px solid var(--line);
        border-radius: 18px;
        background: rgba(255, 250, 242, 0.88);
        backdrop-filter: blur(6px);
        box-shadow: 0 18px 40px rgba(37, 32, 26, 0.08);
      }
      .metric-card {
        padding: 0.85rem 1rem;
        border: 1px solid var(--line);
        border-radius: 14px;
        background: var(--card);
        min-height: 104px;
      }
      .small-label {
        font-size: 0.82rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--muted);
      }
      .big-value {
        font-size: 1.55rem;
        font-weight: 700;
        color: var(--ink);
      }
      .section-card {
        padding: 0.95rem 1rem;
        border: 1px solid var(--line);
        border-radius: 16px;
        background: rgba(255, 250, 242, 0.9);
        margin-bottom: 1rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


KNOWN_METHODS = list(METHOD_SPECS.keys())
DP_METHODS = ["laplace_noise", "embedding_dp_laplace"]
OTHER_METHODS = ["gaussian_noise", "gaussian_blur", "blur_plus_noise", "embedding_noise", "random_projection", "quantization", "cancellable_transform"]
BENCHMARK_CONFIGS = {
    "embedding_noise": {"parameter_name": "sigma", "values": [0.01, 0.03, 0.05, 0.1, 0.15], "base_params": {}},
    "random_projection": {"parameter_name": "target_dim", "values": [512, 256, 128, 64, 32], "base_params": {"seed": 42}},
    "quantization": {"parameter_name": "levels", "values": [256, 128, 64, 32, 16, 8], "base_params": {}},
    "cancellable_transform": {"parameter_name": "mix_ratio", "values": [0.2, 0.4, 0.6, 0.8, 1.0], "base_params": {"seed": 42}},
}
BUNDLED_EXTERNAL_IMAGES = {
    "Greta Thunberg (Wikimedia Commons, not in LFW)": PROJECT_ROOT / "sample_inputs" / "greta_thunberg_unknown.jpg",
}


@st.cache_resource(show_spinner="Loading ArcFace pipeline and cached embeddings...")
def get_pipeline() -> PrivacyFacePipeline:
    return PrivacyFacePipeline.discover(PROJECT_ROOT)


def render_metric_card(label: str, value: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="small-label">{label}</div>
          <div class="big-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_section_header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="section-card">
          <div class="small-label">{title}</div>
          <div style="font-size: 1.05rem; color: #172121; margin-top: 0.35rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_param_inputs(method: str, prefix: str) -> dict[str, float | int]:
    spec = METHOD_SPECS[method]
    params: dict[str, float | int] = {}
    for name, default in spec.default_params.items():
        key = f"{prefix}_{method}_{name}"
        if isinstance(default, int):
            max_value = max(default * 4, 10)
            params[name] = st.slider(name, min_value=1, max_value=max_value, value=default, key=key)
        else:
            max_value = max(float(default) * 4.0, 1.0)
            params[name] = st.slider(
                name,
                min_value=0.01,
                max_value=max_value,
                value=float(default),
                step=0.01,
                key=key,
            )
    return params


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image.convert("RGB"))


def make_heatmap_figure(embedding: np.ndarray, title: str):
    figure, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(build_visual_embedding_map(embedding), cmap="magma")
    axis.set_title(title)
    axis.axis("off")
    return figure


def dataframe_download(df: pd.DataFrame, filename: str) -> tuple[bytes, str]:
    return df.to_csv(index=False).encode("utf-8"), filename


def choose_dataset_test_sample(pipeline: PrivacyFacePipeline, key: str) -> tuple[np.ndarray, str, str]:
    test_frame = pipeline.dataset.frame.iloc[pipeline.test_idx].copy()
    test_frame["display"] = (
        test_frame["label_name"].str.replace("_", " ", regex=False)
        + " | "
        + test_frame["image_path"].apply(lambda path: Path(path).name)
    )
    selected = st.selectbox("Test sample", test_frame["display"].tolist(), key=key)
    row = test_frame.iloc[test_frame["display"].tolist().index(selected)]
    image_rgb = np.asarray(Image.open(row["image_path"]).convert("RGB"))
    return image_rgb, row["label_name"], row["image_path"]


def choose_unknown_source(pipeline: PrivacyFacePipeline, prefix: str) -> tuple[np.ndarray | None, str | None]:
    mode = st.radio(
        "Unknown source",
        ["Bundled external sample", "Excluded LFW identity", "Upload image"],
        key=f"{prefix}_unknown_mode",
    )
    if mode == "Bundled external sample":
        selected_label = st.selectbox("Bundled sample", list(BUNDLED_EXTERNAL_IMAGES.keys()), key=f"{prefix}_bundled")
        path = BUNDLED_EXTERNAL_IMAGES[selected_label]
        return np.asarray(Image.open(path).convert("RGB")), str(path)
    if mode == "Excluded LFW identity":
        unknown_paths = pipeline.sample_unknown_paths(count=40)
        selected_path = st.selectbox("Unknown sample", [str(path) for path in unknown_paths], key=f"{prefix}_unknown_path")
        return np.asarray(Image.open(selected_path).convert("RGB")), selected_path
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key=f"{prefix}_upload")
    if uploaded is None:
        return None, None
    return pil_to_rgb_array(Image.open(uploaded)), uploaded.name


def render_query_result(result: QueryResult) -> None:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(result.query_image, caption="Original image", use_container_width=True)
    with c2:
        if result.protected_image is not None:
            st.image(result.protected_image, caption=f"Protected image ({result.method})", use_container_width=True)
        else:
            st.pyplot(make_heatmap_figure(result.query_embedding, f"{result.method} embedding"), clear_figure=True)
    with c3:
        st.image(str(result.matched_image_path), caption=f"Best match: {result.matched_label.replace('_', ' ')}", use_container_width=True)

    st.json(
        {
            "method": result.method,
            "predicted_label": result.predicted_label,
            "matched_label": result.matched_label,
            "distance": result.predicted_distance,
            "is_known": result.is_known,
            "is_correct": result.is_correct,
        }
    )


def render_method_reference(method: str) -> None:
    note = METHOD_NOTES.get(method)
    if not note:
        return
    with st.expander("Method note", expanded=False):
        st.write(note["how_it_works"])
        st.write(note["utility_privacy"])
        for source in note["sources"]:
            st.markdown(f"- [{source['label']}]({source['url']})")


def compute_unknown_summary(pipeline: PrivacyFacePipeline, method: str, params: dict[str, float | int], threshold: float, count: int) -> tuple[pd.DataFrame, float]:
    unknown_df = pipeline.evaluate_unknown_images(
        image_paths=pipeline.sample_unknown_paths(count=count),
        method=method,
        params=params,
        threshold=threshold,
    )
    rejection_rate = float(unknown_df["rejected"].mean()) if not unknown_df.empty else 0.0
    return unknown_df, rejection_rate


def render_overview_section(pipeline: PrivacyFacePipeline) -> None:
    summary = pipeline.dataset_summary()
    st.markdown('<div class="hero">', unsafe_allow_html=True)
    st.title("Privacy-Preserving Face Recognition")
    st.write(
        "Single GUI for ArcFace, LFW, differential privacy sweeps, embedding anonymization, "
        "known-user matching, unknown-user rejection, and automatic benchmarking."
    )
    st.markdown("</div>", unsafe_allow_html=True)

    cols = st.columns(5)
    values = [
        ("Total identities", str(summary["total_identities"])),
        ("Total images", str(summary["total_images"])),
        ("Filtered identities", str(summary["filtered_identities"])),
        ("Filtered images", str(summary["filtered_images"])),
        ("Default threshold", f'{summary["threshold"]:.3f}'),
    ]
    for col, (label, value) in zip(cols, values):
        with col:
            render_metric_card(label, value)

    render_section_header(
        "Current workflow",
        "Use the sections in the left sidebar to run one task at a time: known users, unknown users, DP sweeps, other anonymization methods, then the automatic benchmark.",
    )

    st.subheader("Dataset preview")
    grid = pipeline.get_sample_paths()
    preview_cols = st.columns(len(grid))
    for col, paths in zip(preview_cols, grid):
        if not paths:
            continue
        col.caption(Path(paths[0]).parent.name.replace("_", " "))
        for path in paths:
            col.image(str(path), use_container_width=True)

    st.subheader("Persistent artifacts")
    st.write(f"Embeddings cache: `{pipeline.config.embeddings_cache_path}`")
    st.write(f"Classifier cache: `{pipeline.config.classifier_cache_path}`")
    st.write(f"Experiments folder: `{pipeline.config.experiments_dir}`")


def render_known_users_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 1",
        "Repeat matching using photos from different users and compute the accuracy of the ML method.",
    )
    left, right = st.columns([1, 2])
    with left:
        st.info("Change the controls, then press the main button. Heavy actions show a spinner and keep the last result visible below.")
        method = st.selectbox("Method", KNOWN_METHODS, key="known_method")
        params = render_param_inputs(method, prefix="known")
        threshold = st.slider("Known-user threshold", 0.0, 1.0, float(pipeline.threshold), 0.01, key="known_threshold")
        limit = st.slider("Number of test samples", 50, len(pipeline.test_idx), min(300, len(pipeline.test_idx)), 10, key="known_limit")
        sample_rgb, known_label, source_name = choose_dataset_test_sample(pipeline, key="known_sample")

        if st.button("Run sample match", key="known_match_button", use_container_width=True):
            with st.spinner("Running sample match..."):
                st.session_state["known_query_result"] = pipeline.query_image(
                    image_rgb=sample_rgb,
                    method=method,
                    params=params,
                    known_label=known_label,
                    threshold=threshold,
                )
                st.session_state["known_query_source"] = source_name

        if st.button("Compute accuracy", key="known_eval_button", type="primary", use_container_width=True):
            with st.spinner("Computing matching accuracy on known users..."):
                result = pipeline.evaluate_method(method=method, params=params, threshold=threshold, limit=limit)
                st.session_state["known_eval_result"] = result
        render_method_reference(method)

    with right:
        result = st.session_state.get("known_query_result")
        if isinstance(result, QueryResult):
            st.write(f"Sample source: `{st.session_state.get('known_query_source')}`")
            render_query_result(result)
        eval_result = st.session_state.get("known_eval_result")
        if eval_result is not None:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                render_metric_card("Top-1 accuracy", f"{eval_result.top1_accuracy:.3f}")
            with c2:
                render_metric_card("Rejection rate", f"{eval_result.rejection_rate:.3f}")
            with c3:
                render_metric_card("Mean distance", f"{eval_result.mean_distance:.3f}")
            with c4:
                render_metric_card("Rejected", f"{eval_result.num_rejected}/{eval_result.num_samples}")
            st.caption("Save accuracy result writes a JSON summary into the experiments folder.")
            if st.button("Save accuracy result", key="known_save_result"):
                saved = pipeline.save_experiment(
                    "known_users_accuracy",
                    {"type": "known_users_accuracy", "result": eval_result.__dict__},
                )
                st.success(f"Saved to {saved}")


def render_unknown_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 2",
        "Use a photo of a person not in the database, or an external image, and check whether it is identified or rejected.",
    )
    left, right = st.columns([1, 2])
    with left:
        st.info("Use this section for a person outside the database or for a truly external image. If the distance is above the threshold, the system should reject it.")
        method = st.selectbox("Method", KNOWN_METHODS, key="unknown_method_main")
        params = render_param_inputs(method, prefix="unknown_main")
        threshold = st.slider("Unknown-user threshold", 0.0, 1.0, float(pipeline.threshold), 0.01, key="unknown_threshold_main")
        image_rgb, source_name = choose_unknown_source(pipeline, prefix="unknown_main")
        batch_count = st.slider("Batch size for excluded LFW users", 5, 40, 20, 5, key="unknown_batch_count")

        if st.button("Check one unknown image", key="unknown_single_button", type="primary", use_container_width=True) and image_rgb is not None:
            with st.spinner("Checking whether the unknown image is matched or rejected..."):
                st.session_state["unknown_single_result"] = pipeline.query_image(
                    image_rgb=image_rgb,
                    method=method,
                    params=params,
                    known_label=None,
                    threshold=threshold,
                )
                st.session_state["unknown_single_source"] = source_name

        if st.button("Run batch rejection test", key="unknown_batch_button", use_container_width=True):
            with st.spinner("Running batch rejection test on unknown identities..."):
                unknown_df, rejection_rate = compute_unknown_summary(pipeline, method, params, threshold, batch_count)
                st.session_state["unknown_batch_df"] = unknown_df
                st.session_state["unknown_batch_rate"] = rejection_rate
        render_method_reference(method)

    with right:
        single_result = st.session_state.get("unknown_single_result")
        if isinstance(single_result, QueryResult):
            st.write(f"Source: `{st.session_state.get('unknown_single_source')}`")
            render_query_result(single_result)
        batch_df = st.session_state.get("unknown_batch_df")
        if isinstance(batch_df, pd.DataFrame):
            render_metric_card("Unknown rejection rate", f"{st.session_state.get('unknown_batch_rate', 0.0):.3f}")
            st.dataframe(batch_df, use_container_width=True)


def render_dp_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 3",
        "Test differential privacy with different epsilon values and compare utility on known users against rejection on unknown users.",
    )
    left, right = st.columns([1, 2])
    with left:
        st.info("Privacy levels here are the epsilon values. Smaller epsilon means stronger privacy and usually lower utility.")
        dp_method = st.radio(
            "DP mode",
            options=DP_METHODS,
            format_func=lambda value: "Image Laplace noise" if value == "laplace_noise" else "Embedding Laplace noise",
            key="dp_method",
        )
        base_params = render_param_inputs(dp_method, prefix="dp")
        threshold = st.slider("DP threshold", 0.0, 1.0, float(pipeline.threshold), 0.01, key="dp_threshold")
        limit = st.slider("Known-user sample count", 50, len(pipeline.test_idx), min(250, len(pipeline.test_idx)), 10, key="dp_limit")
        unknown_count = st.slider("Unknown-user sample count", 5, 40, 20, 5, key="dp_unknown_count")
        raw_values = st.text_input("Epsilon values", "1,2,4,8,16", key="dp_eps_values")

        if st.button("Run DP sweep", key="dp_run_button", type="primary", use_container_width=True):
            with st.spinner("Running differential privacy sweep across epsilon values..."):
                epsilon_values = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
                rows: list[dict[str, float | int | str]] = []
                for epsilon in epsilon_values:
                    params = dict(base_params)
                    params["epsilon"] = epsilon
                    known_result = pipeline.evaluate_method(dp_method, params=params, threshold=threshold, limit=limit)
                    _, unknown_rejection_rate = compute_unknown_summary(pipeline, dp_method, params, threshold, unknown_count)
                    rows.append(
                        {
                            "method": dp_method,
                            "epsilon": epsilon,
                            "known_top1_accuracy": known_result.top1_accuracy,
                            "known_rejection_rate": known_result.rejection_rate,
                            "unknown_rejection_rate": unknown_rejection_rate,
                            "mean_distance": known_result.mean_distance,
                        }
                    )
                st.session_state["dp_sweep_df"] = pd.DataFrame(rows)
        render_method_reference(dp_method)

    with right:
        sweep_df = st.session_state.get("dp_sweep_df")
        if isinstance(sweep_df, pd.DataFrame):
            st.dataframe(sweep_df, use_container_width=True)
            st.line_chart(sweep_df.set_index("epsilon")[["known_top1_accuracy", "unknown_rejection_rate"]])
            data, filename = dataframe_download(sweep_df, "dp_sweep.csv")
            st.download_button("Download DP sweep CSV", data=data, file_name=filename, mime="text/csv")
            st.caption("Save DP sweep writes the epsilon table into the experiments folder.")
            if st.button("Save DP sweep", key="dp_save_button"):
                saved = pipeline.save_experiment("dp_sweep", {"type": "dp_sweep", "rows": sweep_df.to_dict(orient="records")})
                st.success(f"Saved to {saved}")


def render_other_methods_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 4",
        "Choose another anonymization method, select an image, apply the protection, and inspect the visual or embedding-level effect with the retrieved match.",
    )
    left, right = st.columns([1, 2])
    with left:
        st.info("This is the manual single-image demo. Choose one method, one image, and inspect the visual result plus the retrieved identity.")
        method = st.selectbox("Anonymization method", OTHER_METHODS, key="other_method")
        params = render_param_inputs(method, prefix="other")
        threshold = st.slider("Matching threshold", 0.0, 1.0, float(pipeline.threshold), 0.01, key="other_threshold")
        source_mode = st.radio("Image source", ["Dataset test image", "Bundled external sample", "Excluded LFW identity", "Upload image"], key="other_source_mode")
        known_label = None
        source_name = None
        image_rgb = None
        if source_mode == "Dataset test image":
            image_rgb, known_label, source_name = choose_dataset_test_sample(pipeline, key="other_dataset_sample")
        elif source_mode == "Bundled external sample":
            bundled_label = st.selectbox("Bundled sample", list(BUNDLED_EXTERNAL_IMAGES.keys()), key="other_bundled_sample")
            bundled_path = BUNDLED_EXTERNAL_IMAGES[bundled_label]
            image_rgb = np.asarray(Image.open(bundled_path).convert("RGB"))
            source_name = str(bundled_path)
        elif source_mode == "Excluded LFW identity":
            unknown_paths = pipeline.sample_unknown_paths(count=40)
            selected_path = st.selectbox("Excluded LFW sample", [str(path) for path in unknown_paths], key="other_unknown_path")
            image_rgb = np.asarray(Image.open(selected_path).convert("RGB"))
            source_name = selected_path
        else:
            uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"], key="other_upload")
            if uploaded is not None:
                image_rgb = pil_to_rgb_array(Image.open(uploaded))
                source_name = uploaded.name

        if st.button("Run anonymization demo", key="other_run_button", type="primary", use_container_width=True) and image_rgb is not None:
            with st.spinner("Running manual anonymization demo..."):
                st.session_state["other_query_result"] = pipeline.query_image(
                    image_rgb=image_rgb,
                    method=method,
                    params=params,
                    known_label=known_label,
                    threshold=threshold,
                )
                st.session_state["other_query_source"] = source_name
        render_method_reference(method)

    with right:
        result = st.session_state.get("other_query_result")
        if isinstance(result, QueryResult):
            st.write(f"Source: `{st.session_state.get('other_query_source')}`")
            render_query_result(result)
            st.caption("Save anonymization demo writes a JSON summary into the experiments folder.")
            if st.button("Save anonymization demo", key="other_save_button"):
                saved = pipeline.save_experiment(
                    "other_anonymization_demo",
                    {
                        "type": "other_anonymization_demo",
                        "source": st.session_state.get("other_query_source"),
                        "method": result.method,
                        "predicted_label": result.predicted_label,
                        "distance": result.predicted_distance,
                    },
                )
                st.success(f"Saved to {saved}")


def render_benchmark_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 5",
        "Automatic benchmark: run all main anonymization methods once, scan the parameter range, then save the results for later reopening.",
    )
    left, right = st.columns([1, 2])
    with left:
        st.info("This section is different from the manual demo: it automatically sweeps several methods and parameter ranges in one run, then saves the benchmark table.")
        threshold = st.slider("Benchmark threshold", 0.0, 1.0, float(pipeline.threshold), 0.01, key="benchmark_threshold")
        limit = st.slider("Benchmark sample count", 50, len(pipeline.test_idx), min(200, len(pipeline.test_idx)), 10, key="benchmark_limit")
        if st.button("Run full benchmark", key="benchmark_run_button", type="primary", use_container_width=True):
            with st.spinner("Running automatic benchmark across all selected anonymization methods..."):
                frames: list[pd.DataFrame] = []
                for method, config in BENCHMARK_CONFIGS.items():
                    sweep = pipeline.run_parameter_sweep(
                        method=method,
                        parameter_name=config["parameter_name"],
                        parameter_values=config["values"],
                        base_params=config["base_params"],
                        threshold=threshold,
                        limit=limit,
                    )
                    sweep["threshold"] = threshold
                    frames.append(sweep)
                benchmark_df = pd.concat(frames, ignore_index=True)
                st.session_state["benchmark_df"] = benchmark_df

    with right:
        benchmark_df = st.session_state.get("benchmark_df")
        if isinstance(benchmark_df, pd.DataFrame):
            st.dataframe(benchmark_df, use_container_width=True)
            for method in BENCHMARK_CONFIGS:
                method_df = benchmark_df[benchmark_df["method"] == method].copy()
                st.markdown(f"**{method}**")
                st.line_chart(method_df.set_index("parameter_value")[["top1_accuracy", "rejection_rate"]])
            data, filename = dataframe_download(benchmark_df, "automatic_benchmark.csv")
            st.download_button("Download benchmark CSV", data=data, file_name=filename, mime="text/csv")
            st.caption("Save benchmark writes the benchmark table into the experiments folder.")
            if st.button("Save benchmark", key="benchmark_save_button"):
                saved = pipeline.save_experiment(
                    "automatic_benchmark",
                    {"type": "automatic_benchmark", "rows": benchmark_df.to_dict(orient="records")},
                )
                st.success(f"Saved to {saved}")


def render_saved_results_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 6",
        "Reload saved experiments and exported results without recomputing embeddings or retraining the model.",
    )
    paths = pipeline.load_saved_experiments()
    if not paths:
        st.info("No saved experiments yet.")
        return
    selected = st.selectbox("Saved experiment", [str(path) for path in paths], key="saved_select")
    content = json.loads(Path(selected).read_text(encoding="utf-8"))
    st.json(content)


def render_methods_section() -> None:
    render_section_header(
        "Section 7",
        "Method notes for the next meeting: how each anonymization works, what to expect on utility/privacy, and the reference sources.",
    )
    for key, spec in METHOD_SPECS.items():
        with st.expander(f"{spec.name} ({spec.space})", expanded=False):
            st.write(spec.description)
            if spec.default_params:
                st.json(spec.default_params)
            note = METHOD_NOTES.get(key)
            if note:
                st.markdown("**How it works**")
                st.write(note["how_it_works"])
                st.markdown("**Why use it**")
                st.write(note["why_use_it"])
                st.markdown("**Utility vs privacy**")
                st.write(note["utility_privacy"])
                st.markdown("**Sources**")
                for source in note["sources"]:
                    st.markdown(f"- [{source['label']}]({source['url']})")


def main() -> None:
    pipeline = get_pipeline()
    with st.sidebar:
        st.title("Work Sections")
        section = st.radio(
            "Go to",
            [
                "Overview",
                "1. Known Users Accuracy",
                "2. Unknown / OOD Check",
                "3. Differential Privacy Sweeps",
                "4. Other Anonymization Methods",
                "5. Automatic Benchmark",
                "6. Saved Results",
                "7. Method Notes",
            ],
        )
        st.caption("Heavy actions start only when you press a main button. A spinner appears while the computation is running. Save buttons write JSON summaries into the experiments folder.")

    if section == "Overview":
        render_overview_section(pipeline)
    elif section == "1. Known Users Accuracy":
        render_known_users_section(pipeline)
    elif section == "2. Unknown / OOD Check":
        render_unknown_section(pipeline)
    elif section == "3. Differential Privacy Sweeps":
        render_dp_section(pipeline)
    elif section == "4. Other Anonymization Methods":
        render_other_methods_section(pipeline)
    elif section == "5. Automatic Benchmark":
        render_benchmark_section(pipeline)
    elif section == "6. Saved Results":
        render_saved_results_section(pipeline)
    else:
        render_methods_section()


if __name__ == "__main__":
    main()
