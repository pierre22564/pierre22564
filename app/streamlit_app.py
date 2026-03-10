from __future__ import annotations

import sys
import time
import traceback
from datetime import datetime
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
        word-break: break-word;
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


ANONYMIZATION_METHODS = [
    "gaussian_blur",
    "embedding_noise",
    "random_projection",
    "quantization",
    "cancellable_transform",
]

METHOD_LABELS = {
    "gaussian_blur": "Gaussian Blur",
    "embedding_noise": "Noise Injection",
    "random_projection": "Random Projection",
    "quantization": "Quantization",
    "cancellable_transform": "Cancellable Transformation",
    "embedding_dp_laplace": "Differential Privacy (Laplace on embedding)",
}

UI_PARAM_FIELDS = {
    "gaussian_blur": ["kernel_size"],
    "embedding_noise": ["sigma"],
    "random_projection": ["target_dim"],
    "quantization": ["levels"],
    "cancellable_transform": ["mix_ratio"],
    "embedding_dp_laplace": ["epsilon", "sensitivity"],
}

BUNDLED_EXTERNAL_IMAGES = {
    "Greta Thunberg (human, not in LFW)": PROJECT_ROOT / "sample_inputs" / "greta_thunberg_unknown.jpg",
    "Cat image (OOD animal sample)": PROJECT_ROOT / "sample_inputs" / "cat_ood.jpg",
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


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def render_param_inputs(method: str, prefix: str) -> dict[str, float | int]:
    spec = METHOD_SPECS[method]
    params: dict[str, float | int] = {}
    for name in UI_PARAM_FIELDS.get(method, list(spec.default_params.keys())):
        default = spec.default_params[name]
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


def add_overall_success_rate(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if {"top1_accuracy", "rejection_rate"}.issubset(enriched.columns):
        enriched["overall_success_rate"] = enriched["top1_accuracy"] * (1.0 - enriched["rejection_rate"])
    return enriched


def init_ui_state() -> None:
    st.session_state.setdefault("activity_log", [])
    st.session_state.setdefault("current_action", None)
    st.session_state.setdefault("last_error", None)
    st.session_state.setdefault("last_success", None)


def log_activity(level: str, message: str, duration: float | None = None, details: str | None = None) -> None:
    entry = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "message": message,
        "duration": duration,
        "details": details,
    }
    log = st.session_state.get("activity_log", [])
    log.insert(0, entry)
    st.session_state["activity_log"] = log[:20]


def run_ui_action(label: str, action):
    start = time.perf_counter()
    st.session_state["current_action"] = label
    st.session_state["last_error"] = None
    log_activity("info", f"Started: {label}")
    status = st.status(label, expanded=True)
    status.write("Running...")
    try:
        result = action(status)
        duration = time.perf_counter() - start
        status.update(label=f"{label} completed", state="complete", expanded=False)
        status.write(f"Completed in {duration:.2f}s")
        st.session_state["last_success"] = {"label": label, "duration": duration}
        log_activity("success", f"Completed: {label}", duration=duration)
        return result
    except Exception:
        duration = time.perf_counter() - start
        tb = traceback.format_exc()
        st.session_state["last_error"] = {"label": label, "duration": duration, "traceback": tb}
        status.update(label=f"{label} failed", state="error", expanded=True)
        status.write(f"Failed after {duration:.2f}s")
        status.code(tb, language="python")
        log_activity("error", f"Failed: {label}", duration=duration, details=tb)
        st.error(f"{label} failed. See the error details below.")
        with st.expander("Error details", expanded=True):
            st.code(tb, language="python")
        return None
    finally:
        st.session_state["current_action"] = None


def render_activity_console() -> None:
    st.subheader("System Console")
    current_action = st.session_state.get("current_action")
    if current_action:
        st.warning(f"Current action: {current_action}")
    else:
        st.success("Idle")

    last_success = st.session_state.get("last_success")
    if last_success:
        st.caption(f"Last completed action: {last_success['label']} in {last_success['duration']:.2f}s")

    last_error = st.session_state.get("last_error")
    if last_error:
        st.error(f"Last error: {last_error['label']} after {last_error['duration']:.2f}s")
        with st.expander("Last error traceback", expanded=False):
            st.code(last_error["traceback"], language="python")

    for entry in st.session_state.get("activity_log", [])[:8]:
        suffix = f" ({entry['duration']:.2f}s)" if entry["duration"] is not None else ""
        st.caption(f"[{entry['time']}] {entry['level'].upper()} - {entry['message']}{suffix}")


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
        ["Bundled sample", "Excluded LFW identity", "Upload image"],
        key=f"{prefix}_unknown_mode",
    )
    if mode == "Bundled sample":
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
    top_cols = st.columns(4)
    with top_cols[0]:
        render_metric_card("Predicted identity", result.predicted_label.replace("_", " "))
    with top_cols[1]:
        render_metric_card("Nearest label", result.matched_label.replace("_", " "))
    with top_cols[2]:
        render_metric_card("Cosine distance", f"{result.predicted_distance:.3f}")
    with top_cols[3]:
        render_metric_card("Decision", "Matched" if result.is_known else "Rejected")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.image(result.query_image, caption="Original image", use_container_width=True)
    with c2:
        if result.protected_image is not None:
            st.image(result.protected_image, caption=f"Protected image ({method_label(result.method)})", use_container_width=True)
        else:
            st.pyplot(make_heatmap_figure(result.query_embedding, f"{method_label(result.method)} embedding"), clear_figure=True)
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


def compute_unknown_summary(
    pipeline: PrivacyFacePipeline,
    method: str,
    params: dict[str, float | int],
    threshold: float,
    count: int,
) -> tuple[pd.DataFrame, float]:
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
        "This interface is limited to the three requested tasks: known-user matching accuracy, unknown or OOD rejection, "
        "and differential privacy sweeps over epsilon."
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

    method_cols = st.columns(len(ANONYMIZATION_METHODS) + 1)
    method_values = [
        ("Method 1", method_label("gaussian_blur")),
        ("Method 2", method_label("embedding_noise")),
        ("Method 3", method_label("random_projection")),
        ("Method 4", method_label("quantization")),
        ("Method 5", method_label("cancellable_transform")),
        ("DP mode", method_label("embedding_dp_laplace")),
    ]
    for col, (label, value) in zip(method_cols, method_values):
        with col:
            render_metric_card(label, value)

    st.caption(
        f"Embeddings cache: {pipeline.config.embeddings_cache_path} | "
        f"Classifier cache: {pipeline.config.classifier_cache_path} | "
        f"Experiments folder: {pipeline.config.experiments_dir}"
    )


def render_known_users_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 1",
        "Repeat the matching using photos from different users and compute the accuracy of the ML method.",
    )
    left, right = st.columns([1, 2])
    with left:
        st.info("Choose one anonymization method, run one visual example, then compute the accuracy on multiple known users from the dataset.")
        method = st.selectbox(
            "Anonymization method",
            ANONYMIZATION_METHODS,
            format_func=method_label,
            key="known_method",
        )
        params = render_param_inputs(method, prefix="known")
        threshold = st.slider("Known-user threshold", 0.0, 1.0, float(pipeline.threshold), 0.01, key="known_threshold")
        limit = st.slider("Number of test samples", 20, len(pipeline.test_idx), min(20, len(pipeline.test_idx)), 10, key="known_limit")
        sample_rgb, known_label, source_name = choose_dataset_test_sample(pipeline, key="known_sample")

        if st.button("Run sample match", key="known_match_button", use_container_width=True):

            def _run_known_sample(status):
                status.write("Preparing query image...")
                result = pipeline.query_image(
                    image_rgb=sample_rgb,
                    method=method,
                    params=params,
                    known_label=known_label,
                    threshold=threshold,
                )
                status.write("Matching completed.")
                st.session_state["known_query_result"] = result
                st.session_state["known_query_source"] = source_name

            run_ui_action("Known-user sample match", _run_known_sample)

        if st.button("Compute accuracy", key="known_eval_button", type="primary", use_container_width=True):

            def _run_known_eval(status):
                status.write(f"Evaluating {limit} known-user samples...")
                result = pipeline.evaluate_method(method=method, params=params, threshold=threshold, limit=limit)
                status.write(f"Top-1 accuracy: {result.top1_accuracy:.3f}")
                st.session_state["known_eval_result"] = result
                st.session_state["known_eval_payload"] = {
                    "type": "known_users_accuracy",
                    "method": method,
                    "params": params,
                    "threshold": threshold,
                    "result": result.__dict__,
                }

            run_ui_action("Known-user accuracy evaluation", _run_known_eval)

        st.divider()
        st.markdown("**Blur accuracy sweep**")
        st.caption(
            "Run an automatic blur experiment on multiple known users. "
            "The app increases the blur level and plots how the matching accuracy changes."
        )
        blur_values_text = st.text_input(
            "Blur kernel sizes",
            "1,5,11,21,31,41,61,81",
            key="known_blur_values",
        )
        if st.button("Run blur accuracy sweep", key="known_blur_sweep_button", use_container_width=True):

            def _run_blur_sweep(status):
                kernel_sizes = [int(item.strip()) for item in blur_values_text.split(",") if item.strip()]
                kernel_sizes = [value if value % 2 == 1 else value + 1 for value in kernel_sizes]
                status.write(f"Testing Gaussian Blur on {limit} known-user samples...")
                sweep_df = pipeline.run_parameter_sweep(
                    method="gaussian_blur",
                    parameter_name="kernel_size",
                    parameter_values=kernel_sizes,
                    threshold=threshold,
                    limit=limit,
                )
                sweep_df = add_overall_success_rate(sweep_df)
                st.session_state["known_blur_sweep_df"] = sweep_df
                st.session_state["known_blur_sweep_payload"] = {
                    "type": "known_user_blur_sweep",
                    "method": "gaussian_blur",
                    "threshold": threshold,
                    "num_samples": limit,
                    "rows": sweep_df.to_dict(orient="records"),
                }
                status.write("Blur sweep completed.")

            run_ui_action("Known-user blur accuracy sweep", _run_blur_sweep)

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
            st.caption("Save result writes the known-user accuracy summary into the experiments folder.")
            if st.button("Save known-user result", key="known_save_result"):
                saved = pipeline.save_experiment("known_users_accuracy", st.session_state["known_eval_payload"])
                st.success(f"Saved to {saved}")

        blur_sweep_df = st.session_state.get("known_blur_sweep_df")
        if isinstance(blur_sweep_df, pd.DataFrame):
            st.markdown("**Blur sweep results**")
            st.dataframe(blur_sweep_df, use_container_width=True)
            chart_df = blur_sweep_df.set_index("parameter_value")[["overall_success_rate", "top1_accuracy", "rejection_rate"]]
            chart_df.index.name = "kernel_size"
            st.line_chart(chart_df)
            st.caption(
                "Interpretation: `overall_success_rate` is the most important curve here, because it counts rejected images as failures. "
                "As the blur kernel increases, this curve should eventually collapse toward zero."
            )
            csv_data, csv_name = dataframe_download(blur_sweep_df, "known_user_blur_sweep.csv")
            st.download_button(
                "Download blur sweep CSV",
                data=csv_data,
                file_name=csv_name,
                mime="text/csv",
                key="known_blur_download",
            )
            if st.button("Save blur sweep", key="known_blur_save"):
                saved = pipeline.save_experiment("known_user_blur_sweep", st.session_state["known_blur_sweep_payload"])
                st.success(f"Saved to {saved}")


def render_unknown_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 2",
        "Use a photo of a person not in the dataset, or an out-of-distribution sample such as an animal photo, and check whether it is identified or rejected.",
    )
    left, right = st.columns([1, 2])
    with left:
        st.info("This section supports a bundled human unknown sample, a bundled animal OOD sample, excluded LFW identities, or your own upload.")
        method = st.selectbox(
            "Anonymization method",
            ANONYMIZATION_METHODS,
            format_func=method_label,
            key="unknown_method",
        )
        params = render_param_inputs(method, prefix="unknown")
        threshold = st.slider("Unknown-user threshold", 0.0, 1.0, float(pipeline.threshold), 0.01, key="unknown_threshold")
        image_rgb, source_name = choose_unknown_source(pipeline, prefix="unknown")
        batch_count = st.slider("Batch size for excluded LFW users", 5, 40, 20, 5, key="unknown_batch_count")

        if st.button("Check one unknown image", key="unknown_single_button", type="primary", use_container_width=True):
            if image_rgb is None:
                st.warning("Select or upload an image first.")
            else:

                def _run_unknown_single(status):
                    status.write("Computing embedding and nearest match...")
                    result = pipeline.query_image(
                        image_rgb=image_rgb,
                        method=method,
                        params=params,
                        known_label=None,
                        threshold=threshold,
                    )
                    status.write(f"Predicted label: {result.predicted_label}")
                    st.session_state["unknown_single_result"] = result
                    st.session_state["unknown_single_source"] = source_name

                run_ui_action("Unknown single-image check", _run_unknown_single)

        if st.button("Run batch rejection test", key="unknown_batch_button", use_container_width=True):

            def _run_unknown_batch(status):
                status.write(f"Testing {batch_count} unknown samples...")
                unknown_df, rejection_rate = compute_unknown_summary(pipeline, method, params, threshold, batch_count)
                status.write(f"Unknown rejection rate: {rejection_rate:.3f}")
                st.session_state["unknown_batch_df"] = unknown_df
                st.session_state["unknown_batch_rate"] = rejection_rate
                st.session_state["unknown_batch_payload"] = {
                    "type": "unknown_user_check",
                    "method": method,
                    "params": params,
                    "threshold": threshold,
                    "rows": unknown_df.to_dict(orient="records"),
                }

            run_ui_action("Unknown batch rejection test", _run_unknown_batch)

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
            st.caption("Save result writes the unknown-user rejection table into the experiments folder.")
            if st.button("Save unknown-user result", key="unknown_save_button"):
                saved = pipeline.save_experiment("unknown_user_check", st.session_state["unknown_batch_payload"])
                st.success(f"Saved to {saved}")


def render_dp_section(pipeline: PrivacyFacePipeline) -> None:
    render_section_header(
        "Section 3",
        "Test differential privacy with different epsilon values and compare utility on known users against rejection on unknown users.",
    )
    left, right = st.columns([1, 2])
    with left:
        st.info("Smaller epsilon means stronger privacy, more injected Laplace noise, and usually lower recognition performance.")
        dp_method = "embedding_dp_laplace"
        st.caption(f"DP mode used here: {method_label(dp_method)}")
        base_params = render_param_inputs(dp_method, prefix="dp")
        threshold = st.slider("DP threshold", 0.0, 1.0, float(pipeline.threshold), 0.01, key="dp_threshold")
        limit = st.slider("Known-user sample count", 50, len(pipeline.test_idx), min(250, len(pipeline.test_idx)), 10, key="dp_limit")
        unknown_count = st.slider("Unknown-user sample count", 5, 40, 20, 5, key="dp_unknown_count")
        raw_values = st.text_input("Epsilon values", "1,2,4,8,16", key="dp_eps_values")

        if st.button("Run DP sweep", key="dp_run_button", type="primary", use_container_width=True):

            def _run_dp_sweep(status):
                epsilon_values = [float(item.strip()) for item in raw_values.split(",") if item.strip()]
                rows: list[dict[str, float | int | str]] = []
                progress = st.progress(0, text="Starting DP sweep...")
                for index, epsilon in enumerate(epsilon_values, start=1):
                    progress.progress(index / len(epsilon_values), text=f"Testing epsilon = {epsilon}")
                    status.write(f"Running epsilon = {epsilon}")
                    params = dict(base_params)
                    params["epsilon"] = epsilon
                    known_result = pipeline.evaluate_method(dp_method, params=params, threshold=threshold, limit=limit)
                    _, unknown_rejection_rate = compute_unknown_summary(pipeline, dp_method, params, threshold, unknown_count)
                    rows.append(
                        {
                            "epsilon": epsilon,
                            "known_top1_accuracy": known_result.top1_accuracy,
                            "known_rejection_rate": known_result.rejection_rate,
                            "unknown_rejection_rate": unknown_rejection_rate,
                            "mean_distance": known_result.mean_distance,
                        }
                    )
                progress.empty()
                sweep_df = pd.DataFrame(rows)
                st.session_state["dp_sweep_df"] = sweep_df
                st.session_state["dp_sweep_payload"] = {
                    "type": "dp_sweep",
                    "method": dp_method,
                    "base_params": base_params,
                    "threshold": threshold,
                    "rows": sweep_df.to_dict(orient="records"),
                }
                status.write("DP sweep finished.")

            run_ui_action("Differential privacy sweep", _run_dp_sweep)

        render_method_reference(dp_method)

    with right:
        sweep_df = st.session_state.get("dp_sweep_df")
        if isinstance(sweep_df, pd.DataFrame):
            st.dataframe(sweep_df, use_container_width=True)
            st.line_chart(sweep_df.set_index("epsilon")[["known_top1_accuracy", "unknown_rejection_rate"]])
            data, filename = dataframe_download(sweep_df, "dp_sweep.csv")
            st.download_button("Download DP sweep CSV", data=data, file_name=filename, mime="text/csv")
            st.caption("Save result writes the DP epsilon sweep into the experiments folder.")
            if st.button("Save DP sweep", key="dp_save_button"):
                saved = pipeline.save_experiment("dp_sweep", st.session_state["dp_sweep_payload"])
                st.success(f"Saved to {saved}")


def main() -> None:
    init_ui_state()
    try:
        pipeline = get_pipeline()
    except Exception:
        tb = traceback.format_exc()
        st.error("The application failed while loading the pipeline.")
        with st.expander("Startup error details", expanded=True):
            st.code(tb, language="python")
        return

    with st.sidebar:
        st.title("Work Sections")
        section = st.radio(
            "Go to",
            [
                "1. Known Users Accuracy",
                "2. Unknown / OOD Check",
                "3. Differential Privacy Sweeps",
            ],
        )
        st.caption("Only the requested tasks are kept in this GUI. Heavy actions start only when you press a main button.")
        render_activity_console()

    render_overview_section(pipeline)

    if section == "1. Known Users Accuracy":
        render_known_users_section(pipeline)
    elif section == "2. Unknown / OOD Check":
        render_unknown_section(pipeline)
    else:
        render_dp_section(pipeline)


if __name__ == "__main__":
    main()
