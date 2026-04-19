# Privacy-Preserving Face Recognition

Face-recognition project built around ArcFace, LFW, anonymization at the image or embedding level, known-user matching, unknown-user rejection, differential privacy sweeps, and a final utility/privacy conclusion benchmark.

## Documentation Pages

The GitHub-page-ready documentation is split into separate pages inside `docs/`:

- `docs/index.md`
- `docs/project_scope.md`
- `docs/scientific_summary.md`
- `docs/installation.md`
- `docs/results_and_conclusions.md`

## Main entry points

- `app/streamlit_app.py`: global user interface
- `notebooks/01_dataset_arcface_milestone.ipynb`: original notebook
- `docs/anonymization_methods.md`: slide-ready method notes
- `scripts/run_conclusion_benchmark.py`: benchmark used for the final conclusion section
- `scripts/train_arcface_style.py`: dataset-adapted ArcFace-style comparison experiment

## What the app does

The Streamlit GUI is organized into four sections:

- `1. Known Users Accuracy`
  - run one sample match
  - compute matching accuracy on a subset of known users
  - run a blur sweep and plot accuracy versus blur level
- `2. Unknown / OOD Check`
  - test a human image not present in the dataset
  - test an out-of-distribution animal image
  - test excluded LFW identities
- `3. Differential Privacy Sweeps`
  - evaluate Laplace noise on embeddings for several `epsilon` values
  - compare utility on known users and rejection on unknown users
- `4. Conclusion`
  - load the latest benchmark summary
  - show the methodology and tested parameter ranges
  - display utility/privacy plots and the final suggested method
  - compare frozen pre-trained ArcFace with a dataset-adapted ArcFace-style head

Available anonymization methods in the GUI:

- Gaussian blur
- Noise injection
- Differential privacy on embeddings
- Random projection
- Quantization
- Cancellable transformation

## Quick start

### Option 1: one bootstrap command

```bash
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

### Option 2: manual setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Then download:

- LFW archive from `https://ndownloader.figshare.com/files/5976015`
- ArcFace ONNX package from `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`

And place:

- `data/lfw_funneled.tgz` then extract it into `data/lfw_funneled/`
- `models/w600k_r50.onnx`

Then run:

```bash
streamlit run app/streamlit_app.py
```

## Optional Experimental Scripts

Run the conclusion benchmark:

```bash
python3 scripts/run_conclusion_benchmark.py
```

Run the dataset-adapted ArcFace-style comparison:

```bash
python3 scripts/train_arcface_style.py
```

## Bundled external samples

The repo includes two ready-to-use samples for the unknown / OOD section:

- `sample_inputs/greta_thunberg_unknown.jpg`
- `sample_inputs/cat_ood.jpg`

## Persistence

Generated runtime files are stored in:

- `artifacts/`: local caches and local trained adaptation weights
- `experiments/`: saved experiment JSON files
- `results/`: local CSV outputs

Tracked documentation artifacts are stored in:

- `docs/data/`: JSON and CSV-style summaries used by the docs and GUI
- `docs/assets/`: figures used in the GitHub pages and result summaries

These heavy runtime folders are excluded from Git by default, except for the tracked docs files.

## Project structure

- `src/privacy_face_gui/config.py`: paths and cache locations
- `src/privacy_face_gui/data.py`: LFW indexing and image loading
- `src/privacy_face_gui/arcface.py`: ArcFace ONNX embedding extraction
- `src/privacy_face_gui/anonymization.py`: anonymization methods
- `src/privacy_face_gui/method_notes.py`: scientific notes and references
- `src/privacy_face_gui/pipeline.py`: caching, evaluation, matching, persistence
- `docs/`: GitHub-page-ready documentation

## Notes

- The GUI uses cached embeddings and a cached classifier when available.
- If you want a fresh recomputation, delete the files inside `artifacts/`.
- The slide notes for the anonymization methods are in `docs/anonymization_methods.md`.
- The “trained ArcFace” experiment in this repository is a lightweight ArcFace-style adaptation trained on top of frozen ArcFace embeddings. It is not a full retraining of the original large ArcFace backbone.
