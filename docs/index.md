# Privacy-Preserving Face Recognition

This documentation is split into separate pages so the project can be read as a small GitHub page.

## Navigation

- [Project Scope](project_scope.md)
- [Scientific and Technical Summary](scientific_summary.md)
- [Installation Manual](installation.md)
- [Results and Conclusions](results_and_conclusions.md)

## Project Overview

This project studies privacy-preserving face recognition using ArcFace embeddings.

The main idea is:

1. start from a face photo,
2. extract a compact embedding with ArcFace,
3. apply anonymization either on the photo or on the embedding,
4. evaluate the trade-off between recognition utility and privacy.

## Main Deliverables

- a Streamlit application for interactive experiments,
- a dataset of embeddings and labels,
- parameter sweeps for several anonymization methods,
- a comparison between a frozen pre-trained ArcFace pipeline and a dataset-adapted ArcFace-style head,
- experimental conclusions on the utility-privacy trade-off.

## Repository

- Root README: [`README.md`](../README.md)
- App entry point: [`app/streamlit_app.py`](../app/streamlit_app.py)
- Training and benchmark scripts:
  - [`scripts/train_arcface_style.py`](../scripts/train_arcface_style.py)
  - [`scripts/run_conclusion_benchmark.py`](../scripts/run_conclusion_benchmark.py)
