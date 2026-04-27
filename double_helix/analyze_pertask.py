import json, sys
from pathlib import Path
from collections import defaultdict

data = json.load(open(Path(__file__).resolve().parents[1] / "double_helix" / "results" / "multiseed_phi4mini_20260414_083621.json"))

cells = data["cells"]

by_bug_type = defaultdict(lambda: {"single": [], "blind": [], "feedback": []})

for cell in cells:
    cond = cell["condition"]
    seed = cell.get("seed", "?")
    for task in cell["per_task_results"]:
        bt = task["bug_type"]
        solved = task["solved"]
        if cond == "single_shot":
            by_bug_type[bt]["single"].append((seed, solved))
        elif cond == "blind_retry":
            by_bug_type[bt]["blind"].append((seed, solved))
        elif cond == "feedback_retry":
            by_bug_type[bt]["feedback"].append((seed, solved))

print(f"{'Bug Type':25s} | {'Single':>8} | {'Blind':>8} | {'Feedback':>8} | {'Delta':>8}")
print("-" * 70)

for bt in sorted(by_bug_type.keys()):
    d = by_bug_type[bt]
    s_rate = sum(1 for _, v in d["single"] if v) / max(len(d["single"]), 1)
    b_rate = sum(1 for _, v in d["blind"] if v) / max(len(d["blind"]), 1)
    f_rate = sum(1 for _, v in d["feedback"] if v) / max(len(d["feedback"]), 1)
    delta = f_rate - b_rate
    print(f"{bt:25s} | {s_rate:7.1%} | {b_rate:7.1%} | {f_rate:7.1%} | {delta:+7.1%}")

print("\n--- Tasks where feedback helped (blind=0, feedback=1) ---")
helped = defaultdict(int)
hurt = defaultdict(int)
for cell in cells:
    cond = cell["condition"]
    seed = cell.get("seed", "?")
    if cond not in ("blind_retry", "feedback_retry"):
        continue
    for task in cell["per_task_results"]:
        bt = task["bug_type"]
        tid = task["task_id"]
        key = (tid, bt, seed)
        if cond == "blind_retry":
            by_bug_type[bt][f"blind_solved_{seed}"] = task["solved"]
        elif cond == "feedback_retry":
            by_bug_type[bt][f"fb_solved_{seed}"] = task["solved"]

print("\n--- Cross-seed consistency ---")
for bt in sorted(by_bug_type.keys()):
    d = by_bug_type[bt]
    blind_wins = 0
    fb_wins = 0
    ties = 0
    for seed in [42, 123, 456]:
        b = d.get(f"blind_solved_{seed}")
        f = d.get(f"fb_solved_{seed}")
        if b is not None and f is not None:
            if f and not b:
                fb_wins += 1
            elif b and not f:
                blind_wins += 1
            else:
                ties += 1
    if fb_wins > 0 or blind_wins > 0:
        print(f"  {bt:25s}: fb_wins={fb_wins}, blind_wins={blind_wins}, ties={ties}")
