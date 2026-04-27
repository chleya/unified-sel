import os
import sys
import subprocess

sys.path.insert(0, "F:\\unified-sel")
os.chdir("F:\\unified-sel")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from core.capability_benchmark import generate_code_tasks

tasks = generate_code_tasks(num_tasks=1, seed=42, variant="standard")
task = tasks[0]

fn_name = task.metadata.get("function_name", "solve")
buggy_code = task.metadata.get("buggy_code", "")

print(f"=== Task: {fn_name} ===")
print(f"Buggy code:\n{buggy_code}")
print()

prompt = (
    f"Fix the bug in this Python function.\n"
    f"Function: {fn_name}\n"
    f"Tests:\n  solve(5) == 15\n"
    f"Buggy code:\n```python\n{buggy_code}```\n"
    f"Output ONLY the corrected function.\n"
)

gguf_path = r"F:\unified-sel\double_helix\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

cmd = [
    "llama-cli",
    "-m", gguf_path,
    "-p", prompt,
    "-n", "384",
    "-c", "2048",
    "--temp", "0.3",
    "-ngl", "0",
    "--no-warmup",
    "--log-disable",
    "-no-cnv",
    "--single-turn",
    "--simple-io",
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
print(f"=== Raw output ===")
print(result.stdout[-1500:] if len(result.stdout) > 1500 else result.stdout)
