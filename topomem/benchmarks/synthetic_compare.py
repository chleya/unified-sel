"""
Compare baseline vs verify on the SAME 20 arithmetic problems.
"""
import os, sys, json, time, random, re
from pathlib import Path

PROJECT_ROOT = Path(r"F:\unified-sel")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Fixed 20 problems
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
    text = text.strip()
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

def run_baseline(problems, api_key):
    results = []
    for i, item in enumerate(problems):
        try:
            resp = call_llm(item["problem"], api_key)
            correct = verify(resp, item["answer"])
            results.append({"i": i, "problem": item["problem"], "answer": item["answer"], "predicted": resp, "correct": correct})
            print(f"  Baseline {i+1}/20: {'PASS' if correct else 'FAIL'} | {item['problem']} | got: {extract_number(resp)}")
        except Exception as e:
            results.append({"i": i, "problem": item["problem"], "answer": item["answer"], "predicted": str(e), "correct": False})
            print(f"  Baseline {i+1}/20: FAIL ERROR: {e}")
        time.sleep(0.5)
    return results

def run_verify(problems, api_key, max_retries=2):
    results = []
    for i, item in enumerate(problems):
        problem = item["problem"]
        answer = item["answer"]
        all_attempts = []
        
        for attempt in range(max_retries + 1):
            try:
                resp = call_llm(problem, api_key)
                pred_num = extract_number(resp)
                correct = verify(resp, answer)
                all_attempts.append({"attempt": attempt+1, "predicted": resp, "pred_num": pred_num, "correct": correct})
                if correct:
                    break
            except Exception as e:
                all_attempts.append({"attempt": attempt+1, "predicted": str(e), "pred_num": None, "correct": False})
                break
            time.sleep(0.3)
        
        final_correct = any(a["correct"] for a in all_attempts)
        results.append({"i": i, "problem": problem, "answer": answer, "attempts": all_attempts, "correct": final_correct})
        
        status = "PASS" if final_correct else "FAIL"
        n_attempts = len(all_attempts)
        print(f"  Verify {i+1}/20: {status} (attempts: {n_attempts}) | {problem} | {extract_number(all_attempts[-1]['predicted']) if all_attempts else 'ERROR'}")
        time.sleep(0.5)
    
    return results

def main():
    api_key = load_api_key()
    if not api_key:
        print("ERROR: No API key")
        return
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = PROJECT_ROOT / "topomem" / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Run baseline
    print("\n=== BASELINE (no verify, no retry) ===")
    baseline_results = run_baseline(PROBLEMS, api_key)
    baseline_acc = sum(1 for r in baseline_results if r["correct"]) / len(PROBLEMS)
    print(f"\nBaseline: {baseline_acc:.0%} ({sum(1 for r in baseline_results if r['correct'])}/{len(PROBLEMS)})")
    
    # Save baseline
    with open(out_dir / f"synthetic_compare_baseline_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({"mode": "baseline", "accuracy": baseline_acc, "results": baseline_results}, f, indent=2, ensure_ascii=False)
    
    time.sleep(3)
    
    # Run verify
    print("\n=== VERIFY (sympy verify + retry) ===")
    verify_results = run_verify(PROBLEMS, api_key)
    verify_acc = sum(1 for r in verify_results if r["correct"]) / len(PROBLEMS)
    print(f"\nVerify: {verify_acc:.0%} ({sum(1 for r in verify_results if r['correct'])}/{len(PROBLEMS)})")
    
    # Save verify
    with open(out_dir / f"synthetic_compare_verify_{ts}.json", "w", encoding="utf-8") as f:
        json.dump({"mode": "verify", "accuracy": verify_acc, "results": verify_results}, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n=== COMPARISON ===")
    print(f"Baseline: {baseline_acc:.0%}")
    print(f"Verify:   {verify_acc:.0%}")
    print(f"Delta:    {verify_acc - baseline_acc:+.0%}")
    print(f"\nResults saved to {out_dir}")

if __name__ == "__main__":
    main()
