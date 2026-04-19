# Project Scope

[Home](index.md) | [Scientific Summary](scientific_summary.md) | [Installation](installation.md) | [Results](results_and_conclusions.md)

## Scope

Modern face recognition systems rely on compact vector embeddings rather than raw images.
These embeddings are efficient and discriminative, but they can still behave like stable biometric identifiers.

The project goal is to design and evaluate embedding-level and image-level anonymization methods for face authentication.

## Research Question

For a given recognition utility level, which anonymization method offers the strongest privacy protection?

## Experimental Pipeline

1. input face image
2. ArcFace embedding extraction
3. anonymization method applied either before or after ArcFace
4. nearest-neighbor matching in the gallery
5. evaluation on known users, unknown users, and out-of-distribution samples

## Dataset

The project uses the **LFW funneled** dataset.

- full dataset: `5,749` identities and `13,233` images
- filtered subset used in experiments: `62` identities and `3,023` images
- filtering rule: at least `20` images per identity

## Practical Deliverables

- an interactive Streamlit GUI,
- reproducible benchmark scripts,
- saved plots and result tables,
- GitHub documentation pages with installation steps and conclusions.
