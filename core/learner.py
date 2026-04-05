"""
Unified classifier wrapper around StructurePool.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from core.pool import StructurePool


class UnifiedSELClassifier:
    def __init__(
        self,
        in_size: int,
        out_size: int,
        lr: float = 0.05,
        max_structures: int = 20,
        evolve_every: int = 20,
        pool_config: Optional[Dict] = None,
        seed: Optional[int] = None,
    ):
        self.in_size = in_size
        self.out_size = out_size
        self.lr = lr
        self.evolve_every = evolve_every
        self.step = 0

        pool_kwargs = dict(pool_config or {})
        pool_kwargs["max_structures"] = max_structures
        self.pool = StructurePool(
            in_size=in_size,
            out_size=out_size,
            initial_structures=1,
            seed=seed,
            **pool_kwargs,
        )

        rng = np.random.default_rng(seed)
        self.W_out = rng.normal(0.0, 0.1, size=(out_size, out_size))
        self._history: List[Dict] = []

    def fit_one(self, x: np.ndarray, y: np.ndarray) -> float:
        x = x.flatten()
        y = np.atleast_1d(y).flatten()

        observation = self.pool.observe(x)
        active = observation["active_structure"]
        effective_lr = self.lr

        hidden = active.forward(x)
        output = (hidden @ self.W_out).flatten()

        if y.shape[0] == 1:
            target = np.zeros(self.out_size)
            target[int(y[0])] = 1.0
        else:
            target = y

        error = target - output
        loss = self.pool.learn_active(
            active_structure=active,
            x=np.atleast_2d(x),
            output_error=np.atleast_2d(error),
            lr=effective_lr,
        )

        self.W_out += effective_lr * np.outer(hidden.flatten(), error)
        self.W_out = np.clip(self.W_out, -2.0, 2.0)

        self.step += 1
        if self.step % self.evolve_every == 0:
            self.pool.evolve()

        self._history.append(
            {
                "step": self.step,
                "loss": float(loss),
                "event": observation["event"],
                "surprise": float(observation["surprise"]),
                "n_structures": int(observation["n_structures"]),
            }
        )
        return float(loss)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        x = x.flatten()
        hidden = self.pool.forward(np.atleast_2d(x))
        output = (hidden @ self.W_out).flatten()
        output -= output.max()
        exp_out = np.exp(output)
        return exp_out / (exp_out.sum() + 1e-8)

    def predict(self, x: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(x)))

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        correct = sum(1 for idx in range(len(X)) if self.predict(X[idx]) == int(y[idx]))
        return correct / len(X)

    def get_stats(self) -> Dict:
        stats = self.pool.get_stats()
        stats["total_steps"] = self.step
        if self._history:
            recent = self._history[-20:]
            stats["recent_avg_loss"] = float(np.mean([row["loss"] for row in recent]))
            stats["recent_avg_surprise"] = float(np.mean([row["surprise"] for row in recent]))
        return stats

    def get_event_counts(self) -> Dict:
        counts = {"reinforce": 0, "branch": 0, "create": 0}
        for row in self._history:
            event = row.get("event", "")
            if event in counts:
                counts[event] += 1
        return counts
