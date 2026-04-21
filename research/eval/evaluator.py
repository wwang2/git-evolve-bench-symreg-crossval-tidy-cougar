"""Evaluator for the symbolic-regression-with-crossval benchmark.

Contract:
    python3 research/eval/evaluator.py --solution <path> --seed <int>
    stdout: METRIC=<float>

Metric: mean squared error of `f(x)` on a clean held-out test set of 500 points
drawn uniformly from [-5, 5]. Minimize. Target 0.01.

Agents are expected to use `research/eval/train.npz` (200 noisy training points,
sigma=0.05) to infer `f(x)`. The target function below is deliberately
concealed — please do not read it; treat this benchmark as true symbolic
regression from noisy data.
"""

from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent


def _target(x: np.ndarray) -> np.ndarray:
    # Hidden target: agents should recover this from noisy training data.
    return np.sin(x) + 0.3 * x


def _test_points(seed: int, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    x = rng.uniform(-5.0, 5.0, size=n)
    x.sort()
    y = _target(x)
    return x, y


def _load_solution(path: str):
    path = pathlib.Path(path).resolve()
    spec = importlib.util.spec_from_file_location("solution", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["solution"] = module
    spec.loader.exec_module(module)
    return module


def evaluate(solution_path: str, seed: int = 42) -> float:
    x_test, y_test = _test_points(seed=seed)
    module = _load_solution(solution_path)
    if not hasattr(module, "f"):
        raise AttributeError("solution.py must define function f(x)")
    y_pred = np.asarray(module.f(x_test), dtype=float).ravel()
    if y_pred.shape != y_test.shape:
        raise ValueError(
            f"f(x) returned shape {y_pred.shape}, expected {y_test.shape}"
        )
    return float(np.mean((y_pred - y_test) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    metric = evaluate(args.solution, args.seed)
    print(f"METRIC={metric}")


if __name__ == "__main__":
    main()
