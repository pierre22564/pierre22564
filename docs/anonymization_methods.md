# Anonymization Methods for the Next Meeting

This file is structured so each section can be converted directly into one slide.

## Differential Privacy / Laplace Noise

- Core idea: add calibrated random noise controlled by the privacy budget `epsilon`.
- In this project: implemented as Laplace perturbation on image pixels and on ArcFace embeddings.
- Main intuition: smaller `epsilon` means stronger privacy and lower expected utility.
- What to show:
  - original image or embedding
  - perturbed version for several `epsilon` values
  - matching accuracy vs `epsilon`
- Sources:
  - [The Algorithmic Foundations of Differential Privacy](https://www.cis.upenn.edu/~aaroth/privacybook.html)
  - [Differential Privacy tutorial](https://privacytools.seas.harvard.edu/files/privacytools/files/differential_privacy_tutorial_-_salil_vadhan_-_2014-06-101.pdf)

## Random Projection

- Core idea: project embeddings with a seeded random matrix into a lower-dimensional space.
- Why it matters: distances can remain usable while the original embedding becomes harder to recover.
- In this project: implemented as a seeded projection with configurable target dimension.
- What to show:
  - original embedding
  - projected embedding
  - accuracy vs target dimension
- Sources:
  - [Cancelable templates for secure face verification based on deep learning and random projections](https://link.springer.com/article/10.1186/s13635-023-00147-y)
  - [An Analysis of Random Projections in Cancelable Biometrics](https://arxiv.org/abs/1401.4489)

## Quantization

- Core idea: reduce embedding precision by mapping continuous values to a small number of discrete levels.
- Why it matters: lower precision leaks less detail and can support binary or compact protected templates.
- In this project: implemented as uniform quantization with configurable number of levels.
- What to show:
  - original embedding values
  - quantized embedding values
  - accuracy vs number of quantization levels
- Sources:
  - [Binary Biometric Representation through Pairwise Adaptive Phase Quantization](https://link.springer.com/article/10.1155/2011/543106)
  - [Secure and Efficient Biometric-Data Binarization using Multi-Objective Optimization](https://link.springer.com/article/10.1080/18756891.2015.1113746)

## Cancellable Transformations

- Core idea: apply a user- or system-specific keyed transform so the protected template can be revoked and regenerated.
- Why it matters: unlike a raw biometric, a compromised protected template does not need to remain permanent.
- In this project: implemented as a seeded permutation and sign-flip transform.
- What to show:
  - original embedding
  - transformed embedding
  - same face under two different seeds gives two different protected templates
- Sources:
  - [Cancelable biometrics: A case study in fingerprints](https://research.ibm.com/publications/cancelable-biometrics-a-case-study-in-fingerprints)
  - [Enhancing security and privacy in biometrics-based authentication systems](https://research.ibm.com/publications/enhancing-security-and-privacy-in-biometrics-based-authentication-systems)
  - [Biometric Template Protection for Neural-Network-based Face Recognition Systems: A Survey](https://arxiv.org/abs/2110.05044)
