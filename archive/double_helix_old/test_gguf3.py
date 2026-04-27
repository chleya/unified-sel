import os, sys, subprocess, winreg

sys.path.insert(0, "F:\\unified-sel")
os.chdir("F:\\unified-sel")

key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
machine_path, _ = winreg.QueryValueEx(key, "Path")
key2 = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
user_path, _ = winreg.QueryValueEx(key2, "Path")
os.environ["PATH"] = machine_path + ";" + user_path

from core.capability_benchmark import generate_code_tasks
from double_helix.boundary_scan import _extract_code, _strip_imports

tasks = generate_code_tasks(num_tasks=1, seed=42, variant="standard")
task = tasks[0]
fn_name = task.metadata.get("function_name", "solve")
buggy_code = task.metadata.get("buggy_code", "")

user_msg = (
    f"Fix the bug in this Python function.\n"
    f"Function: {fn_name}\n"
    f"Tests:\n  solve(5) == 15\n"
    f"Buggy code:\n```python\n{buggy_code}```\n"
    f"Output ONLY the corrected function.\n"
)

prompt = f"<|system|>You are a Python code fixer. Output ONLY the corrected function.<|end|>\n<|user|>\n{user_msg}<|end|>\n<|assistant| >\n"

gguf_path = r"F:\unified-sel\double_helix\microsoft_Phi-4-mini-instruct-Q4_K_M.gguf"

cmd = [
    "llama-cli", "-m", gguf_path, "-p", prompt,
    "-n", "384", "-c", "2048", "--temp", "0.3",
    "-ngl", "0", "--no-warmup", "--log-disable",
    "-no-cnv", "--single-turn", "--simple-io", "--no-display-prompt",
]

result = subprocess.run(cmd, capture_output=True, timeout=120)
response = result.stdout.decode("utf-8", errors="replace").strip()
for tok in ["<|end|>", "<|assistant| >", "<start_of_turn>", "<end_of_turn>"]:
    response = response.replace(tok, "")

with open(r"F:\unified-sel\double_helix\test_output.txt", "w", encoding="utf-8") as f:
    f.write(f"=== RAW RESPONSE ===\n{response}\n\n")
    code = _extract_code(response)
    f.write(f"=== EXTRACTED CODE ===\n{code}\n\n")
    code = _strip_imports(code)
    f.write(f"=== AFTER STRIP IMPORTS ===\n{code}\n")

print("Done - check test_output.txt")
