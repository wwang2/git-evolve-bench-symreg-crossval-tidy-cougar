"""Baseline solution: predicts the training-mean for every x.

MSE against the clean held-out test set should be ~ variance of f_true
over [-5, 5] — roughly 1.2. Used for the `baselines.baseline` anchor in
research/config.yaml and for `validate_baseline.py`.
"""

from __future__ import annotations

import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_TRAIN = _HERE / "train.npz"
_Y_MEAN: float | None = None


def _y_mean() -> float:
    global _Y_MEAN
    if _Y_MEAN is None:
        data = np.load(_TRAIN)
        _Y_MEAN = float(np.mean(data["y"]))
    return _Y_MEAN


def f(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.full_like(x, _y_mean())
