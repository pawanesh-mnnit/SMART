"""
tools/imports.py — common imports and small shared helpers.

Every script in SMART starts with `from tools.imports import *` so that the
scientific-stack imports, the device string and the shape/range check helper
are defined in exactly one place.
"""

import argparse
import csv
import json
import os
import pickle
import random
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fnn
    import torchvision as tv
    from torchvision import transforms
    from PIL import Image

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TORCH_AVAILABLE = True
except ImportError:  # evaluate.py / dataset.py are pure-numpy and must still work
    torch = nn = Fnn = tv = transforms = Image = None
    DEVICE = "cpu"
    TORCH_AVAILABLE = False

try:
    from tqdm.auto import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

warnings.filterwarnings("ignore", category=UserWarning)
np.set_printoptions(precision=3, suppress=True)

__all__ = [
    "argparse", "csv", "json", "os", "pickle", "random", "sys", "Path",
    "np", "pd", "torch", "nn", "Fnn", "tv", "transforms", "Image",
    "DEVICE", "TORCH_AVAILABLE", "tqdm",
    "set_seed", "ck", "banner",
]


def set_seed(seed: int = 42) -> None:
    """Seed python / numpy / torch so a run is reproducible."""
    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def ck(name, a):
    """Print shape / dtype / range of an array — the sanity print used throughout."""
    if hasattr(a, "shape") and getattr(a, "size", 0):
        print(f"  [check] {name}: shape={tuple(a.shape)} dtype={a.dtype} "
              f"min={np.min(a):.3f} max={np.max(a):.3f}")
    else:
        print(f"  [check] {name}: {a}")


def banner(text: str) -> None:
    print("\n" + "=" * 72)
    print(text)
    print("=" * 72)
