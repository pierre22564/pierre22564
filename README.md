# Privacy-Preserving Face Recognition

Face-recognition project built around ArcFace, LFW, image degradation, embedding anonymization, unknown-user rejection, and automatic benchmarking.

## Main entry points

- `app/streamlit_app.py`: global user interface
- `notebooks/01_dataset_arcface_milestone.ipynb`: original notebook
- `docs/anonymization_methods.md`: slide-ready method notes

## What the app does

- Browse LFW samples
- Match known users and compute accuracy
- Test unknown or out-of-distribution images
- Sweep differential privacy levels with different `epsilon` values
- Try other anonymization methods:
  - noise injection
  - random projection
  - quantization
  - cancellable transformations
- Run an automatic benchmark across several methods
- Save and reopen experiment results

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

## External sample image

The repo includes one external sample image not present in the local LFW dataset:

- `sample_inputs/greta_thunberg_unknown.jpg`

This is useful for the unknown-user section of the GUI.

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
