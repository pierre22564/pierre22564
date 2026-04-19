# Installation Manual

[Home](index.md) | [Project Scope](project_scope.md) | [Scientific Summary](scientific_summary.md) | [Results](results_and_conclusions.md)

## Requirements

- Python `3.12`
- `pip`
- enough disk space for:
  - LFW dataset
  - ArcFace ONNX model
  - cached embeddings and experimental results

## Quick Installation

```bash
git clone https://github.com/pierre22564/pierre22564.git
cd pierre22564
chmod +x scripts/bootstrap.sh
./scripts/bootstrap.sh
source .venv/bin/activate
streamlit run app/streamlit_app.py
```

## Manual Installation

### 1. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Download the data and model

Download:

- LFW archive: `https://ndownloader.figshare.com/files/5976015`
- ArcFace model package: `https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip`

Place:

- `data/lfw_funneled.tgz`
- `models/w600k_r50.onnx`

Then extract the LFW archive into:

- `data/lfw_funneled/`

### 3. Launch the application

```bash
streamlit run app/streamlit_app.py
```

## Optional Experimental Scripts

### Conclusion benchmark

```bash
python3 scripts/run_conclusion_benchmark.py
```

### ArcFace-style adaptation comparison

```bash
python3 scripts/train_arcface_style.py
```

## Output Folders

- `artifacts/`: local caches and trained local adaptation weights
- `experiments/`: saved experiment JSON files
- `results/`: local CSV outputs
- `docs/data/`: tracked summary JSON used by the docs and the conclusion page
- `docs/assets/`: tracked figures used by the docs
