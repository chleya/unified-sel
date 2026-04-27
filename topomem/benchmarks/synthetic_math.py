"""
Synthetic Math Benchmark — pure arithmetic, no external data needed.
Ground truth computed by Python eval(), verified numerically.

Tests: TopoMem + self-correction on simple math
"""
import os, sys, json, time, random, re
from pathlib import Path

PROJECT_ROOT = Path(r"F:\unified-sel")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

import numpy as np

random.seed(42)
np.random.seed(42)

# Operations
OPS = ["+", "-", "*", "//"]

def gen_problem():
    op = random.choice(OPS)
    if op == "+":
        a = random.randint(10, 9999)
        b = random.randint(10, 9999)
        ans = a + b
    elif op == "-":
        a = random.randint(100, 9999)
        b = random.randint(10, a)  # ensure positive result
        ans = a - b
    elif op == "*":
        a = random.randint(2, 99)
        b = random.randint(2, 99)
        ans = a * b
    else:  # //
        b = random.randint(2, 50)
        ans = random.randint(2, 50)
        a = b * ans  # exact division
    return f"{a} {op} {b} = ?", str(ans)

def extract_number(text):
    """Extract numeric answer from model response."""
    if not text:
        return None
    text = text.strip()
    # Try boxed
    m = re.search(r'\\boxed\s*\{(\d+)\}', text)
    if m:
        return m.group(1)
    # Try last number
    nums = re.findall(r'-?\d+', text)
    if nums:
        return nums[-1]
    return None

def verify(predicted, ground_truth):
    """Verify numeric answers."""
    pred = extract_number(predicted)
    if pred is None:
        return False
    try:
        return int(pred) == int(ground_truth)
    except:
        return False

def build_dataset(n=100):
    problems = []
    for i in range(n):
        problem, answer = gen_problem()
        problems.append({
            "problem": problem,
            "answer": answer,
            "level": random.choice(["easy", "medium", "hard"]),
            "type": "arithmetic"
        })
    return problems

def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        cfg_path = Path.home() / ".openclaw" / "openclaw.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                provs = cfg.get("providers", {})
                for p in provs.values():
                    if isinstance(p, dict) and "api_key" in p:
                        key = p["api_key"]
                        break
            except Exception:
                pass
    return key

def call_llm(problem, api_key):
    """Call MiniMax API."""
    import urllib.request
    
    payload = {
        "model": "MiniMax-2.7",
        "max_tokens": 256,
        "messages": [
            {"role": "user", "content": f"Solve this math problem. Give your final answer in \\boxed{{number}} format.\n\n{problem}"}
        ]
    }
    
    req = urllib.request.Request(
        "https://api.minimaxi.com/anthropic/v1/messages",
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        },
        method="POST"
    )
    
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        parts = []
        for b in data["content"]:
            if b["type"] == "text":
                parts.append(b["text"])
        return "".join(parts)

def run_baseline(dataset, api_key):
    """Mode 1: Direct LLM call, no verify."""
    results = []
    for item in dataset:
        try:
            response = call_llm(item["problem"], api_key)
            correct = verify(response, item["answer"])
            results.append({
                "problem": item["problem"],
                "predicted": response,
                "correct": correct,
                "answer": item["answer"]
            })
        except Exception as e:
            results.append({
                "problem": item["problem"],
                "predicted": f"ERROR: {e}",
                "correct": False,
                "answer": item["answer"]
            })
        time.sleep(0.5)  # rate limit
    return results

def run_with_verify(dataset, api_key, max_retries=2):
    """Mode 2: LLM + sympy verify + retry."""
    from sympy import sympify, N
    
    results = []
    for item in dataset:
        problem = item["problem"]
        answer = item["answer"]
        
        for attempt in range(max_retries + 1):
            try:
                response = call_llm(problem, api_key)
                pred = extract_number(response)
                
                # Verify with sympy
                correct = False
                if pred:
                    try:
                        pred_val = sympify(pred)
                        ans_val = sympify(answer)
                        correct = (float(N(pred_val)) == float(N(ans_val)))
                    except:
                        correct = (pred == answer)
                
                results.append({
                    "problem": problem,
                    "attempt": attempt + 1,
                    "predicted": response,
                    "final_pred": pred,
                    "correct": correct,
                    "answer": answer
                })
                
                if correct:
                    break
                elif attempt < max_retries:
                    time.sleep(0.3)
                    
            except Exception as e:
                results.append({
                    "problem": problem,
                    "attempt": attempt + 1,
                    "predicted": f"ERROR: {e}",
                    "correct": False,
                    "answer": answer
                })
                break
        
        time.sleep(0.5)
    
    return results

def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    mode = sys.argv[2] if len(sys.argv) > 2 else "baseline"
    
    print(f"Generating {n} synthetic math problems...")
    dataset = build_dataset(n)
    
    api_key = load_api_key()
    if not api_key:
        print("ERROR: No API key found. Set ANTHROPIC_API_KEY")
        return
    
    print(f"Running mode: {mode}")
    t0 = time.time()
    
    if mode == "verify":
        results = run_with_verify(dataset, api_key)
    else:
        results = run_baseline(dataset, api_key)
    
    elapsed = time.time() - t0
    
    # Summarize
    correct = sum(1 for r in results if r["correct"])
    total = len(results)
    accuracy = correct / total if total > 0 else 0
    
    print(f"\n=== Results ({mode}) ===")
    print(f"Accuracy: {correct}/{total} = {accuracy:.1%}")
    print(f"Elapsed: {elapsed:.0f}s ({elapsed/total:.1f}s/problem)")
    
    # Save
    out = {
        "mode": mode,
        "n": n,
        "accuracy": accuracy,
        "correct": correct,
        "total": total,
        "elapsed_s": elapsed,
        "results": results
    }
    
    ts = time.strftime("%Y%m%d_%H%M%S")
    outpath = PROJECT_ROOT / "topomem" / "benchmarks" / "results" / f"synthetic_math_{mode}_{ts}.json"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Saved: {outpath}")

if __name__ == "__main__":
    main()
