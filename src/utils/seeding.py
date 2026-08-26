"""Deterministic seeding across Python, NumPy, and Torch.

Called once at the top of the notebook and at the start of every experiment script.
Reproducibility is a graded requirement, and an unseeded dense retriever or a
non-deterministic batch order would make the Task 6 comparison unfair.
"""

from __future__ import annotations

import os
import random


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Pin every RNG we depend on. Returns the seed for logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            # Inference-only workload, so the throughput cost of disabling the
            # autotuner is irrelevant, and identical inputs must give identical logits.
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

    return seed
