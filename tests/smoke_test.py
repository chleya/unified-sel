from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.learner import UnifiedSELClassifier
from core.pool import StructurePool
from core.structure import make_structure
from experiments.baselines.ewc import EWCBaseline
from experiments.baselines.fixed import FixedNetwork


def test_structure_creation() -> None:
    s = make_structure(0, in_size=4, out_size=2, label="smoke", rng=np.random.default_rng(0))
    assert s.weights.shape == (4, 2)
    assert s.feedback.shape == (2, 2)
    assert s.label == "smoke"
    print("[OK] Structure 创建")


def test_structure_pool_observe() -> None:
    pool = StructurePool(in_size=4, out_size=2, seed=0)
    result = pool.observe(np.array([1.0, 0.0, 0.0, 0.0]))
    assert result["event"] in {"reinforce", "branch", "create"}
    assert result["n_structures"] >= 1
    print("[OK] StructurePool observe")


def test_unified_sel_forward() -> None:
    clf = UnifiedSELClassifier(in_size=4, out_size=2, seed=0)
    loss = clf.fit_one(np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0]))
    probs = clf.predict_proba(np.array([1.0, 0.0, 0.0, 0.0]))
    assert loss >= 0.0
    assert probs.shape == (2,)
    assert np.isclose(probs.sum(), 1.0, atol=1e-6)
    print("[OK] DFALearner forward")


def test_ewc_baseline() -> None:
    model = EWCBaseline(in_size=4, out_size=2, seed=0)
    loss = model.fit_one(np.array([0.0, 1.0, 0.0, 0.0]), 1)
    model.consolidate()
    pred = model.predict(np.array([0.0, 1.0, 0.0, 0.0]))
    assert loss >= 0.0
    assert pred in {0, 1}
    print("[OK] EWC baseline")


def test_fixed_baseline() -> None:
    model = FixedNetwork(in_size=4, out_size=2, seed=0)
    loss = model.fit_one(np.array([0.0, 0.0, 1.0, 0.0]), 0)
    pred = model.predict(np.array([0.0, 0.0, 1.0, 0.0]))
    assert loss >= 0.0
    assert pred in {0, 1}
    print("[OK] Fixed baseline")


def main() -> None:
    test_structure_creation()
    test_structure_pool_observe()
    test_unified_sel_forward()
    test_ewc_baseline()
    test_fixed_baseline()
    print("所有测试通过")


if __name__ == "__main__":
    main()
