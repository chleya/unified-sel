import os
import sys

sys.path.insert(0, "F:\\unified-sel")
os.chdir("F:\\unified-sel")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from double_helix.boundary_scan import _try_gguf
from core.capability_benchmark import generate_code_tasks

tasks = generate_code_tasks(num_tasks=1, seed=42, variant="standard")
task = tasks[0]

gguf_path = r"F:\unified-sel\double_helix\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"
code, vis, all_p, fb = _try_gguf(task, gguf_path)

print(f"Vis: {vis}, All: {all_p}, FB: {fb}")
if code:
    print(f"Code snippet: {code[:300]}")
else:
    print("EMPTY code")
