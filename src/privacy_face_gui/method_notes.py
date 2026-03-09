from __future__ import annotations


METHOD_NOTES = {
    "gaussian_noise": {
        "how_it_works": "Adds zero-mean Gaussian noise directly to image pixels. This is a simple degradation baseline rather than a formal privacy guarantee.",
        "why_use_it": "Useful for mapping the recognition break-point as image quality decreases.",
        "utility_privacy": "Usually degrades utility smoothly. Privacy improves only as long as the face becomes hard to match.",
        "sources": [
            {
                "label": "Face template protection survey",
                "url": "https://arxiv.org/abs/2110.05044",
            }
        ],
    },
    "laplace_noise": {
        "how_it_works": "Adds Laplace-distributed noise on the image as a DP-style proxy. In strict differential privacy, epsilon controls the privacy budget.",
        "why_use_it": "Lets you sweep epsilon and observe the utility-privacy trade-off.",
        "utility_privacy": "Smaller epsilon means more noise and usually more privacy, but lower recognition accuracy.",
        "sources": [
            {
                "label": "The Algorithmic Foundations of Differential Privacy",
                "url": "https://www.cis.upenn.edu/~aaroth/privacybook.html",
            },
            {
                "label": "Differential Privacy tutorial",
                "url": "https://privacytools.seas.harvard.edu/files/privacytools/files/differential_privacy_tutorial_-_salil_vadhan_-_2014-06-101.pdf",
            },
        ],
    },
    "gaussian_blur": {
        "how_it_works": "Applies low-pass filtering that removes high-frequency facial detail.",
        "why_use_it": "Simple visual anonymization baseline. Easy to explain and useful in demos.",
        "utility_privacy": "Strong blur improves privacy but quickly destroys recognition utility.",
        "sources": [
            {
                "label": "Face template protection survey",
                "url": "https://arxiv.org/abs/2110.05044",
            }
        ],
    },
    "blur_plus_noise": {
        "how_it_works": "Combines blur and additive noise to stress the recognition pipeline.",
        "why_use_it": "Gives a stronger image-space privacy baseline than blur or noise alone.",
        "utility_privacy": "Can reach higher privacy faster, but often at a steep cost in utility.",
        "sources": [
            {
                "label": "Face template protection survey",
                "url": "https://arxiv.org/abs/2110.05044",
            }
        ],
    },
    "embedding_noise": {
        "how_it_works": "Adds Gaussian noise directly to the embedding vector after ArcFace.",
        "why_use_it": "A direct baseline for template perturbation at the embedding level.",
        "utility_privacy": "Easy to implement. Privacy gain depends on the noise scale and is empirical rather than formal.",
        "sources": [
            {
                "label": "Face template protection survey",
                "url": "https://arxiv.org/abs/2110.05044",
            }
        ],
    },
    "embedding_dp_laplace": {
        "how_it_works": "Adds Laplace noise directly to the embedding. Epsilon tunes the privacy budget.",
        "why_use_it": "Closest in this project to a DP-style embedding anonymization experiment.",
        "utility_privacy": "Stronger privacy with lower epsilon; utility typically drops as the embedding moves away from the original identity cluster.",
        "sources": [
            {
                "label": "The Algorithmic Foundations of Differential Privacy",
                "url": "https://www.cis.upenn.edu/~aaroth/privacybook.html",
            }
        ],
    },
    "random_projection": {
        "how_it_works": "Projects the embedding into a lower-dimensional random subspace using a seeded matrix.",
        "why_use_it": "Often used in cancelable biometrics because the transform can be renewed by changing the seed or projection matrix.",
        "utility_privacy": "Can preserve useful distances surprisingly well while increasing irreversibility, but the choice of projection dimension matters.",
        "sources": [
            {
                "label": "Cancelable templates for secure face verification based on deep learning and random projections",
                "url": "https://link.springer.com/article/10.1186/s13635-023-00147-y",
            },
            {
                "label": "An Analysis of Random Projections in Cancelable Biometrics",
                "url": "https://arxiv.org/abs/1401.4489",
            },
        ],
    },
    "quantization": {
        "how_it_works": "Rounds continuous embedding coordinates into a limited number of discrete levels.",
        "why_use_it": "Reduces information precision and can support secure binary or low-bit protected templates.",
        "utility_privacy": "Moderate quantization can keep matching usable; aggressive quantization improves privacy but may collapse discriminative detail.",
        "sources": [
            {
                "label": "Binary Biometric Representation through Pairwise Adaptive Phase Quantization",
                "url": "https://link.springer.com/article/10.1155/2011/543106",
            },
            {
                "label": "Secure and Efficient Biometric-Data Binarization using Multi-Objective Optimization",
                "url": "https://link.springer.com/article/10.1080/18756891.2015.1113746",
            },
        ],
    },
    "cancellable_transform": {
        "how_it_works": "Applies a keyed transform to the template so that compromised templates can be revoked and replaced by a new transform.",
        "why_use_it": "This is the canonical biometric-template-protection idea when revocability matters.",
        "utility_privacy": "Good cancelable transforms aim to preserve utility while improving revocability, unlinkability, and non-invertibility.",
        "sources": [
            {
                "label": "Cancelable biometrics: A case study in fingerprints",
                "url": "https://research.ibm.com/publications/cancelable-biometrics-a-case-study-in-fingerprints",
            },
            {
                "label": "Enhancing security and privacy in biometrics-based authentication systems",
                "url": "https://research.ibm.com/publications/enhancing-security-and-privacy-in-biometrics-based-authentication-systems",
            },
            {
                "label": "Biometric Template Protection for Neural-Network-based Face Recognition Systems: A Survey",
                "url": "https://arxiv.org/abs/2110.05044",
            },
        ],
    },
}
