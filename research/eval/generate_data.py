"""One-shot script that writes research/eval/train.npz.

Training data: 200 points (x, y) with x ~ U[-5, 5] and
y = f_true(x) + N(0, 0.05^2). Run once to produce the committed file.
"""

from __future__ import annotations

import pathlib

import numpy as np

from evaluator import _target

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "train.npz"
SEED = 12345
SIGMA = 0.05
N_TRAIN = 200


def main() -> None:
    rng = np.random.default_rng(SEED)
    x = rng.uniform(-5.0, 5.0, size=N_TRAIN)
    x.sort()
    y = _target(x) + rng.normal(0.0, SIGMA, size=N_TRAIN)
    np.savez(OUT, x=x, y=y)
    print(f"Wrote {OUT}  shape={x.shape}  sigma={SIGMA}")


if __name__ == "__main__":
    main()
