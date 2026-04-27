"""
H2-Gated Retry Gate Experiment

Tests: Does H2 predict which problems need retry?
- Baseline: no retry
- Always-verify: retry every problem
- H2-gated: retry only when H2 > threshold

Measures: accuracy, retry count, cost savings
"""
import os, sys, json, time, re, random
from pathlib import Path

PROJECT_ROOT = Path(r"F:\unified-sel")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Same 20 problems as synthetic_compare.py
PROBLEMS = [
    {"problem": "419 + 4516 = ?", "answer": "4935"},
    {"problem": "3757 - 581 = ?", "answer": "3176"},
    {"problem": "8945 + 1434 = ?", "answer": "10379"},
    {"problem": "12 // 4 = ?", "answer": "3"},
    {"problem": "3592 + 3821 = ?", "answer": "7413"},
    {"problem": "9205 + 3267 = ?", "answer": "12472"},
    {"problem": "480 // 16 = ?", "answer": "30"},
    {"problem": "2 * 99 = ?", "answer": "198"},
    {"problem": "7024 - 2797 = ?", "answer": "4227"},
    {"problem": "21 * 29 = ?", "answer": "609"},
    {"problem": "15 * 13 = ?", "answer": "195"},
    {"problem": "192 // 8 = ?", "answer": "24"},
    {"problem": "79 * 35 = ?", "answer": "2765"},
    {"problem": "7537 + 8795 = ?", "answer": "16332"},
    {"problem": "6211 + 1301 = ?", "answer": "7512"},
    {"problem": "82 * 81 = ?", "answer": "6642"},
    {"problem": "75 * 26 = ?", "answer": "1950"},
    {"problem": "760 + 3743 = ?", "answer": "4503"},
    {"problem": "12 * 31 = ?", "answer": "372"},
    {"problem": "6237 + 4564 = ?", "answer": "10801"},
]

def extract_number(text):
    if not text:
        return None
    m = re.search(r'\\boxed\s*\{(\d+)\}', text)
    if m:
        return m.group(1)
    nums = re.findall(r'-?\d+', text)
    return nums[-1] if nums else None

def verify(predicted, ground_truth):
    pred = extract_number(predicted)
    if pred is None:
        return False
    try:
        return int(pred) == int(ground_truth)
    except:
        return False

def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        cfg_path = Path.home() / ".openclaw" / "openclaw.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for p in cfg.get("providers", {}).values():
                    if isinstance(p, dict) and "api_key" in p:
                        key = p["api_key"]
                        break
            except Exception:
                pass
    return key

def call_llm(problem, api_key):
    import urllib.request
    payload = {
        "model": "MiniMax-2.7",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": f"Solve. Answer in \\boxed{{number}} format.\n\n{problem}"}]
    }
    req = urllib.request.Request(
        "https://api.minimaxi.com/anthropic/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "anthropic-version": "2023-06-01"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        parts = [b["text"] for b in data["content"] if b["type"] == "text"]
        return "".join(parts)


def compute_h2_for_problem(topomem_system, problem_text):
    """Compute H2 value for a single problem embedding."""
    from topomem.embedding import EmbeddingManager
    from topomem.topology import TopologyEngine, TopologyConfig

    emb_mgr = EmbeddingManager()
    emb = emb_mgr.encode(problem_text)
    emb_2d = emb.reshape(1, -1)

    topo_cfg = TopologyConfig(max_homology_dim=2, metric="cosine")
    topo_engine = TopologyEngine(config=topo_cfg)
    result = topo_engine.compute_persistence(emb_2d)

    h2_pairs = []
    if len(result) > 2 and len(result[2]) > 0:
        h2_pairs = result[2]

    betti_2 = len(h2_pairs)
    if betti_2 > 0:
        persistences = [p[1] - p[0] for p in h2_pairs]
        mean_pers = sum(persistences) / len(persistences)
        max_pers = max(persistences)
    else:
        mean_pers = 0.0
        max_pers = 0.0

    return {
        "betti_2": betti_2,
        "mean_persistence": mean_pers,
        "max_persistence": max_pers,
    }


def main():
    api_key = load_api_key()
    if not api_key:
        print("ERROR: No API key found")
        return

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "topomem" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("H2-GATED RETRY GATE EXPERIMENT")
    print("=" * 60)

    # PHASE 1: Run baseline (no retry) for all problems
    print("\n[Phase 1] Baseline: no retry for any problem")
    baseline_results = []
    for i, item in enumerate(PROBLEMS):
        try:
            resp = call_llm(item["problem"], api_key)
            correct = verify(resp, item["answer"])
            pred_num = extract_number(resp)
        except Exception as e:
            resp = str(e)
            correct = False
            pred_num = None
        r = {
            "i": i, "problem": item["problem"], "answer": item["answer"],
            "attempt_1": resp, "correct": correct, "pred_num": pred_num
        }
        baseline_results.append(r)
        status = "PASS" if correct else "FAIL"
        print(f"  {i+1:2d}/20: {status} | {item['problem']} | got: {pred_num}")
        time.sleep(0.5)

    baseline_acc = sum(1 for r in baseline_results if r["correct"]) / len(PROBLEMS)
    baseline_retries = 0
    print(f"\nBaseline: {baseline_acc:.0%} ({sum(1 for r in baseline_results if r['correct'])}/{len(PROBLEMS)})")

    # PHASE 2: Run H2 computation for each problem (individually)
    print("\n[Phase 2] Computing H2 for each problem...")
    h2_results = []
    for i, item in enumerate(PROBLEMS):
        h2 = compute_h2_for_problem(None, item["problem"])
        h2_results.append({"i": i, "problem": item["problem"], **h2})
        print(f"  {i+1:2d}/20: betti_2={h2['betti_2']}, mean_pers={h2['mean_persistence']:.4f}, max_pers={h2['max_persistence']:.4f}")
        time.sleep(0.1)

    # Analyze H2 distribution
    betti_2_list = [r["betti_2"] for r in h2_results]
    max_pers_list = [r["max_persistence"] for r in h2_results]
    mean_pers_list = [r["mean_persistence"] for r in h2_results]

    print(f"\nH2 distribution:")
    print(f"  Betti-2: min={min(betti_2_list)}, max={max(betti_2_list)}, unique={sorted(set(betti_2_list))}")
    print(f"  Max persistence: min={min(max_pers_list):.4f}, max={max(max_pers_list):.4f}")
    print(f"  Mean persistence: min={min(mean_pers_list):.4f}, max={max(mean_pers_list):.4f}")

    # PHASE 3: Test different H2 thresholds
    print("\n[Phase 3] Testing H2-gated retry policies...")

    # For always-verify: retry ALL failed problems
    always_verify_results = []
    always_verify_retries = 0
    always_verify_correct = 0
    for r in baseline_results:
        if r["correct"]:
            always_verify_results.append({**r, "attempt_2": None, "final_correct": True, "retried": False})
            always_verify_correct += 1
        else:
            always_verify_retries += 1
            # Try once more
            try:
                resp2 = call_llm(PROBLEMS[r["i"]]["problem"], api_key)
                correct2 = verify(resp2, PROBLEMS[r["i"]]["answer"])
            except Exception as e:
                resp2 = str(e)
                correct2 = False
            always_verify_results.append({**r, "attempt_2": resp2, "final_correct": correct2, "retried": True})
            if correct2:
                always_verify_correct += 1
            time.sleep(0.5)

    always_verify_acc = always_verify_correct / len(PROBLEMS)
    print(f"  Always-verify: {always_verify_acc:.0%} (retries={always_verify_retries}, recovered={always_verify_retries - (len(PROBLEMS) - always_verify_correct)})")

    # Test different H2 thresholds
    # Try using betti_2 >= X as gate
    thresholds_to_test = [0, 1, 2, 3, 4, 5]
    gate_results = []

    for thresh in thresholds_to_test:
        retries_triggered = 0
        gated_correct = 0
        gated_recovered = 0
        gated_details = []

        for r, h2 in zip(baseline_results, h2_results):
            h2_val = h2["betti_2"]
            should_retry = h2_val >= thresh

            if r["correct"]:
                gated_correct += 1
                gated_details.append({**r, "h2": h2_val, "gate_triggered": False, "final_correct": True})
            else:
                if should_retry:
                    retries_triggered += 1
                    try:
                        resp2 = call_llm(PROBLEMS[r["i"]]["problem"], api_key)
                        correct2 = verify(resp2, PROBLEMS[r["i"]]["answer"])
                    except Exception as e:
                        resp2 = str(e)
                        correct2 = False
                    recovered = 1 if correct2 else 0
                    gated_recovered += recovered
                    gated_correct += correct2
                    gated_details.append({**r, "h2": h2_val, "gate_triggered": True, "attempt_2": resp2, "final_correct": correct2, "recovered": recovered})
                    time.sleep(0.5)
                else:
                    # Below threshold, don't retry
                    gated_correct += 0  # stays wrong
                    gated_details.append({**r, "h2": h2_val, "gate_triggered": False, "final_correct": False, "recovered": 0})

        gated_acc = gated_correct / len(PROBLEMS)
        gate_results.append({
            "threshold": thresh,
            "accuracy": gated_acc,
            "retries_triggered": retries_triggered,
            "recovered": gated_recovered,
            "cost_savings": 1.0 - (retries_triggered / max(always_verify_retries, 1))
        })
        print(f"  Thresh>={thresh}: acc={gated_acc:.0%}, retries={retries_triggered}, recovered={gated_recovered}, savings={1.0-(retries_triggered/max(always_verify_retries,1)):.0%}")

    # Also test: always retry (baseline failed ones)
    print(f"\n  [baseline only] ACC={baseline_acc:.0%}, retries=0")

    # Find best gate
    best = max(gate_results, key=lambda x: x["accuracy"])
    print(f"\nBest H2-gated policy: threshold >= {best['threshold']} => {best['accuracy']:.0%} (retries={best['retries_triggered']})")

    # COMPREHENSIVE SUMMARY
    print("\n" + "=" * 60)
    print("COMPREHENSIVE SUMMARY")
    print("=" * 60)
    print(f"Baseline (no retry):          ACC={baseline_acc:.0%}, RETRIES=0")
    print(f"Always-verify:               ACC={always_verify_acc:.0%}, RETRIES={always_verify_retries}, recovered={always_verify_retries-(len(PROBLEMS)-always_verify_correct)}")
    print(f"Best H2-gated:               ACC={best['accuracy']:.0%}, RETRIES={best['retries_triggered']}, savings={best['cost_savings']:.0%}")

    print(f"\n--- H2 Per-Problem Data ---")
    print(f"{'#':>2} | {'Problem':>25} | {'B2':>3} | {'MaxP':>6} | {'Baseline':>8} | {'H2-Gate':>8}")
    print("-" * 65)
    for i, (r, h2) in enumerate(zip(baseline_results, h2_results)):
        gate = "RETRY" if not r["correct"] else "---"
        print(f"{i+1:2d} | {r['problem'][:25]:>25} | {h2['betti_2']:3d} | {h2['max_persistence']:6.4f} | {'PASS' if r['correct'] else 'FAIL':>8} | {gate:>8}")

    # Save results
    output = {
        "timestamp": ts,
        "n_problems": len(PROBLEMS),
        "baseline": {
            "accuracy": baseline_acc,
            "correct": sum(1 for r in baseline_results if r["correct"]),
            "failed": [r["i"] for r in baseline_results if not r["correct"]]
        },
        "always_verify": {
            "accuracy": always_verify_acc,
            "retries": always_verify_retries,
            "recovered": always_verify_retries - (len(PROBLEMS) - always_verify_correct)
        },
        "h2_gated": gate_results,
        "best_gate": best,
        "h2_per_problem": [{"i": r["i"], "problem": r["problem"], **h2} for r, h2 in zip(baseline_results, h2_results)],
        "baseline_results": baseline_results
    }
    out_file = out_dir / f"h2_gated_retry_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
