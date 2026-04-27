"""Debug: verify dual-path readout actually changes predictions."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.learner import UnifiedSELClassifier
import numpy as np

clf = UnifiedSELClassifier(in_size=4, out_size=2, seed=42)

X = np.random.randn(20, 4)
y = np.array([0, 1] * 10)

for i in range(100):
    clf.fit_one(X[i % 20], y[i % 20])

acc_before = clf.accuracy(X, y)
print(f"Accuracy before dual-path: {acc_before:.4f}")
print(f"dual_path_active: {clf.dual_path_active}")
print(f"W_out_memory is None: {clf.W_out_memory is None}")

proba_before = clf.predict_proba(X[0])
print(f"Proba before: {proba_before}")

clf.activate_dual_path(alpha=0.9)
print(f"\nAfter activate_dual_path(alpha=0.9):")
print(f"dual_path_active: {clf.dual_path_active}")
print(f"W_out_memory is None: {clf.W_out_memory is None}")

proba_after = clf.predict_proba(X[0])
print(f"Proba after: {proba_after}")

acc_after = clf.accuracy(X, y)
print(f"Accuracy after dual-path: {acc_after:.4f}")

for i in range(50):
    new_X = np.random.randn(4) + 5
    clf.fit_one(new_X, np.array([1]))

acc_after_drift = clf.accuracy(X, y)
print(f"\nAccuracy after drift (no dual-path would forget): {acc_after_drift:.4f}")
print(f"dual_path_active: {clf.dual_path_active}")
