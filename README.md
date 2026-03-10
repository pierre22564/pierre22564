# Privacy-Preserving Face Recognition

Face-recognition project built around ArcFace, LFW, anonymization at the image or embedding level, known-user matching, unknown-user rejection, and differential privacy sweeps.

## Main entry points

- `app/streamlit_app.py`: global user interface
- `notebooks/01_dataset_arcface_milestone.ipynb`: original notebook
- `docs/anonymization_methods.md`: slide-ready method notes

## What the app does

The Streamlit GUI is intentionally limited to the three requested tasks:

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

Available anonymization methods in the GUI:

- Gaussian blur
- Noise injection
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

## Bundled external samples

The repo includes two ready-to-use samples for the unknown / OOD section:

- `sample_inputs/greta_thunberg_unknown.jpg`
- `sample_inputs/cat_ood.jpg`

## Persistence

Generated runtime files are stored in:

- `artifacts/`: embeddings cache and classifier cache
- `experiments/`: saved experiment JSON files
- `results/`: exported result tables or plots

These folders are excluded from Git by default.

## Project structure

- `src/privacy_face_gui/config.py`: paths and cache locations
- `src/privacy_face_gui/data.py`: LFW indexing and image loading
- `src/privacy_face_gui/arcface.py`: ArcFace ONNX embedding extraction
- `src/privacy_face_gui/anonymization.py`: anonymization methods
- `src/privacy_face_gui/method_notes.py`: scientific notes and references
- `src/privacy_face_gui/pipeline.py`: caching, evaluation, matching, persistence

## Notes

- The GUI uses cached embeddings and a cached classifier when available.
- If you want a fresh recomputation, delete the files inside `artifacts/`.
- The slide notes for the anonymization methods are in `docs/anonymization_methods.md`.
