# Scientific and Technical Summary

[Home](index.md) | [Project Scope](project_scope.md) | [Installation](installation.md) | [Results](results_and_conclusions.md)

## Scientific Summary

The project compares anonymization methods that are applied either:

- **before ArcFace**, directly on the photo,
- **after ArcFace**, on the embedding vector.

This distinction is important because photo-space transformations affect the image before feature extraction, while embedding-space transformations directly protect the biometric template.

## Method Grouping

| Method | Applied to | Parameter controlling anonymization level | Interpretation |
|---|---|---|---|
| Gaussian Blur | Photo, before ArcFace | `kernel_size` | Larger kernel means stronger blur |
| Noise Injection | Embedding, after ArcFace | `sigma` | Larger sigma means stronger perturbation |
| Differential Privacy (Laplace) | Embedding, after ArcFace | `epsilon` | Smaller epsilon means stronger privacy |
| Random Projection | Embedding, after ArcFace | `target_dim` | Lower target dimension means stronger compression |
| Quantization | Embedding, after ArcFace | `levels` | Fewer levels mean stronger discretization |
| Cancellable Transformation | Embedding, after ArcFace | `mix_ratio` | Larger mix ratio means stronger transformation |

## Short Scientific Description of the Methods

### Gaussian Blur

- Applied directly to the face image.
- Removes high-frequency facial details before ArcFace sees the input.
- Easy to explain visually, but it may quickly reduce recognition utility.

### Noise Injection

- Applied directly to the embedding.
- Adds Gaussian perturbation after ArcFace extraction.
- Simple baseline for template perturbation.

### Differential Privacy

- Implemented as Laplace noise on the embedding.
- The privacy parameter is `epsilon`.
- This is the only method in this project with a formal privacy parameter.

### Random Projection

- Projects the embedding into another lower-dimensional space.
- Keeps useful geometry approximately intact while changing the template representation.

### Quantization

- Reduces the precision of embedding values.
- Keeps only coarse information about the face template.

### Cancellable Transformation

- Applies a seeded transformation to the embedding.
- If a template is compromised, it can be changed by changing the seed or configuration.

## Technical Summary

The implemented system includes:

- a pre-trained ArcFace ONNX embedding extractor,
- a labeled embedding gallery built from LFW,
- nearest-neighbor matching with cosine distance,
- evaluation on known users, unknown users, and OOD samples,
- a Streamlit application for interactive experiments,
- benchmark scripts for conclusion and model comparison,
- GitHub documentation pages for project communication.

## Note on the “trained ArcFace” comparison

The repository now includes a **dataset-adapted ArcFace-style head** trained on top of the frozen pre-trained ArcFace embeddings from the LFW subset.

This is a lightweight adaptation experiment that is feasible on local CPU and small data.
It is **not** a full retraining of the original large ArcFace backbone from scratch.

This distinction is important when discussing the results with scientific rigor.
