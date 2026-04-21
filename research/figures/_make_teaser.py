"""Generate research/figures/teaser.png — the noisy training data."""

from __future__ import annotations

import pathlib

import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TRAIN = HERE.parent / "eval" / "train.npz"
OUT = HERE / "teaser.png"


def main() -> None:
    data = np.load(TRAIN)
    x, y = data["x"], data["y"]

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10})
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    ax.scatter(x, y, s=16, alpha=0.75, color="#2f6feb", edgecolor="none",
               label="200 noisy training points (σ=0.05)")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Symbolic regression target: recover f(x) from noisy data")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
