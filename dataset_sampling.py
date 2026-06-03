import os
import random
from typing import List, Tuple

from dataloader import read_samples_from_txt


Sample = Tuple[str, str]


def load_trainval_samples(trainval_txt: str) -> List[Sample]:
    """
    Load all samples from trainval txt.

    Returns:
        List of (image_path, mask_path)
    """
    return read_samples_from_txt(trainval_txt)


def sample_fraction_samples(
    samples: List[Sample],
    fraction: float,
    seed: int = 42,
    nested: bool = True
) -> List[Sample]:
    """
    Sample a fraction of the full trainval sample list.

    Args:
        samples: full sample list
        fraction: e.g. 0.25, 0.5, 0.75, 1.0
        seed: random seed for reproducibility
        nested:
            True  -> smaller fractions are nested subsets of larger ones
            False -> each fraction sampled independently

    Returns:
        subset sample list
    """
    if not (0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    total = len(samples)
    n_subset = max(1, int(round(total * fraction)))

    if nested:
        shuffled = samples.copy()
        rng = random.Random(seed)
        rng.shuffle(shuffled)
        subset = shuffled[:n_subset]
    else:
        rng = random.Random(seed + int(fraction * 1000))
        subset = rng.sample(samples, n_subset)

    return subset


def save_subset_txt(samples: List[Sample], save_path: str) -> None:
    """
    Save a subset sample list to txt in:
        image_path mask_path
    format.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        for image_path, mask_path in samples:
            f.write(f"{image_path} {mask_path}\n")


def fraction_to_tag(fraction: float) -> str:
    """
    Convert fraction 0.25 -> '25'
    """
    return str(int(round(fraction * 100)))


if __name__ == "__main__":
    trainval_txt = "../dataset/train_val.txt"
    samples = load_trainval_samples(trainval_txt)

    for frac in [0.25, 0.5, 0.75, 1.0]:
        subset = sample_fraction_samples(samples, frac, seed=42, nested=True)
        print(f"Fraction {frac:.2f}: {len(subset)} samples")