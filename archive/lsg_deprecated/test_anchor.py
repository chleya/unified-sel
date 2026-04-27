"""Quick test: verify anchor regularization works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.structure import make_structure
import numpy as np

s = make_structure(structure_id=0, in_size=4, out_size=3)
s.age = 100
s.set_anchor()
print("anchor_set:", s.anchor_set)
print("anchor shape:", s.anchor.shape)

X = np.random.randn(10, 4)
y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
s.estimate_anchor_fisher(X, y, 2)
print("fisher shape:", s.anchor_fisher.shape if s.anchor_fisher is not None else None)
print("fisher mean:", s.anchor_fisher.mean() if s.anchor_fisher is not None else None)

pen = s.anchor_penalty(1.0)
print("penalty norm (no drift):", np.linalg.norm(pen))

original_weights = s.weights.copy()
s.weights += 0.1 * np.random.randn(*s.weights.shape)
pen2 = s.anchor_penalty(1.0)
print("penalty norm (after drift):", np.linalg.norm(pen2))

print("\nAnchor regularization is working!" if np.linalg.norm(pen2) > np.linalg.norm(pen) else "ERROR: anchor not working")
