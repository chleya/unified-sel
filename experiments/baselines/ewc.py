"""
EWC baseline for the Phase 2 continual-learning benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.runtime import get_results_path, save_json, timestamp


@dataclass
class EWCBaseline:
    """Linear classifier with a simple diagonal EWC penalty."""

    in_size: int
    out_size: int
    lr: float = 0.05
    ewc_lambda: float = 40.0
    seed: Optional[int] = None
    weights: np.ndarray = field(init=False, repr=False)
    fisher: np.ndarray = field(init=False, repr=False)
    anchor: np.ndarray = field(init=False, repr=False)
    ewc_active: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        self.weights = rng.normal(0.0, 0.1, size=(self.in_size, self.out_size))
        self.fisher = np.zeros_like(self.weights)
        self.anchor = self.weights.copy()

    def fit_one(self, x: np.ndarray, y: np.ndarray | int) -> float:
        x = np.asarray(x).flatten()
        target = self._target_vector(y)
        logits = x @ self.weights
        error = target - logits

        task_grad = np.outer(x, error)
        if self.ewc_active:
            penalty_grad = self.ewc_lambda * self.fisher * (self.anchor - self.weights)
        else:
            penalty_grad = np.zeros_like(task_grad)
        self.weights += self.lr * (task_grad + penalty_grad)
        self.weights = np.clip(self.weights, -2.0, 2.0)
        return float(np.mean(error ** 2))

    def fit_dataset(self, X: np.ndarray, y: np.ndarray, epochs: int = 6) -> float:
        losses = []
        for _ in range(epochs):
            for i in range(len(X)):
                losses.append(self.fit_one(X[i], y[i]))
        return float(np.mean(losses)) if losses else 0.0

    def estimate_fisher(self, X: np.ndarray, y: np.ndarray) -> None:
        fisher = np.zeros_like(self.weights)
        for i in range(len(X)):
            x = np.asarray(X[i]).flatten()
            target = self._target_vector(y[i])
            logits = x @ self.weights
            error = target - logits
            fisher += np.outer(x, error) ** 2
        fisher /= max(len(X), 1)
        self.fisher = np.clip(fisher, 0.0, 5.0)

    def consolidate(
        self,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
    ) -> None:
        """Freeze the current weights as the protected anchor."""
        if X is not None and y is not None:
            self.estimate_fisher(X, y)
        self.anchor = self.weights.copy()
        self.ewc_active = True

    def predict(self, x: np.ndarray) -> int:
        x = np.asarray(x).flatten()
        logits = x @ self.weights
        return int(np.argmax(logits))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for i in range(len(X)) if self.predict(X[i]) == int(y[i]))
        return correct / len(X)

    def _target_vector(self, y: np.ndarray | int) -> np.ndarray:
        y_arr = np.asarray(y).flatten()
        if y_arr.shape[0] == self.out_size:
            return y_arr.astype(float)
        target = np.zeros(self.out_size, dtype=float)
        target[int(y_arr[0])] = 1.0
        return target


def make_task(task_id: int, n_samples: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(0.0, 1.0, size=(n_samples, 4))
    boundary = X[:, 0] + X[:, 1]
    if task_id == 0:
        y = (boundary > 0.0).astype(int)
    elif task_id == 1:
        y = (boundary < 0.0).astype(int)
    else:
        raise ValueError(f"Unsupported task_id: {task_id}")
    return X, y


def run_experiment(seed: int = 7) -> dict:
    model = EWCBaseline(in_size=4, out_size=2, lr=0.05, ewc_lambda=40.0, seed=seed)

    train_task_0 = make_task(task_id=0, n_samples=256, seed=seed)
    test_task_0 = make_task(task_id=0, n_samples=256, seed=seed + 1)
    train_task_1 = make_task(task_id=1, n_samples=256, seed=seed + 2)
    test_task_1 = make_task(task_id=1, n_samples=256, seed=seed + 3)

    loss_task_0 = model.fit_dataset(*train_task_0)
    task_0_after_task_0 = model.accuracy(*test_task_0)
    model.consolidate(*train_task_0)

    loss_task_1 = model.fit_dataset(*train_task_1)
    task_0_after_task_1 = model.accuracy(*test_task_0)
    task_1_after_task_1 = model.accuracy(*test_task_1)

    result = {
        "baseline": "ewc",
        "seed": seed,
        "task_0_accuracy_after_task_0": task_0_after_task_0,
        "task_0_accuracy_after_task_1": task_0_after_task_1,
        "task_1_accuracy_after_task_1": task_1_after_task_1,
        "forgetting_task_0": task_0_after_task_0 - task_0_after_task_1,
        "mean_training_loss": {
            "task_0": loss_task_0,
            "task_1": loss_task_1,
        },
    }

    results_dir = get_results_path("baseline_ewc")
    output_path = results_dir / f"{timestamp()}.json"
    save_json(result, output_path)
    result["saved_to"] = str(output_path)
    return result


def main() -> None:
    result = run_experiment()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
