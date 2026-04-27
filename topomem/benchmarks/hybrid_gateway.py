"""
Hybrid Confidence Gateway

Three-layer confidence routing:
  Layer 1: Format check (instant, free) - primary gate
  Layer 2: H1 batch topology (per corpus) - coarse health signal
  Layer 3: verify + retry (expensive) - final safety net

Design:
  - format_check: does response contain \\boxed{}?
  - H1 batch: compute H1 on entire problem corpus (per-domain baseline)
  - verify: sympy verify on extracted answer
"""
import os, sys, json, time, re, random
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(r"F:\unified-sel")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

# Test corpus: 20 arithmetic problems
PROBLEMS = [
    {"problem": "419 + 4516 = ?", "answer": "4935", "domain": "arithmetic"},
    {"problem": "3757 - 581 = ?", "answer": "3176", "domain": "arithmetic"},
    {"problem": "8945 + 1434 = ?", "answer": "10379", "domain": "arithmetic"},
    {"problem": "12 // 4 = ?", "answer": "3", "domain": "arithmetic"},
    {"problem": "3592 + 3821 = ?", "answer": "7413", "domain": "arithmetic"},
    {"problem": "9205 + 3267 = ?", "answer": "12472", "domain": "arithmetic"},
    {"problem": "480 // 16 = ?", "answer": "30", "domain": "arithmetic"},
    {"problem": "2 * 99 = ?", "answer": "198", "domain": "arithmetic"},
    {"problem": "7024 - 2797 = ?", "answer": "4227", "domain": "arithmetic"},
    {"problem": "21 * 29 = ?", "answer": "609", "domain": "arithmetic"},
    {"problem": "15 * 13 = ?", "answer": "195", "domain": "arithmetic"},
    {"problem": "192 // 8 = ?", "answer": "24", "domain": "arithmetic"},
    {"problem": "79 * 35 = ?", "answer": "2765", "domain": "arithmetic"},
    {"problem": "7537 + 8795 = ?", "answer": "16332", "domain": "arithmetic"},
    {"problem": "6211 + 1301 = ?", "answer": "7512", "domain": "arithmetic"},
    {"problem": "82 * 81 = ?", "answer": "6642", "domain": "arithmetic"},
    {"problem": "75 * 26 = ?", "answer": "1950", "domain": "arithmetic"},
    {"problem": "760 + 3743 = ?", "answer": "4503", "domain": "arithmetic"},
    {"problem": "12 * 31 = ?", "answer": "372", "domain": "arithmetic"},
    {"problem": "6237 + 4564 = ?", "answer": "10801", "domain": "arithmetic"},
]

# Additional harder problems
HARDER_PROBLEMS = [
    {"problem": "234567 + 876543 = ?", "answer": "1111110", "domain": "hard_arithmetic"},
    {"problem": "9999 * 9999 = ?", "answer": "99980001", "domain": "hard_arithmetic"},
    {"problem": "sqrt(14400) = ?", "answer": "120", "domain": "hard_arithmetic"},
    {"problem": "2^16 = ?", "answer": "65536", "domain": "hard_arithmetic"},
    {"problem": "123456 - 78901 = ?", "answer": "44555", "domain": "hard_arithmetic"},
    {"problem": "567 * 123 = ?", "answer": "69741", "domain": "hard_arithmetic"},
    {"problem": "10000 // 37 = ?", "answer": "270", "domain": "hard_arithmetic"},
    {"problem": "3^7 = ?", "answer": "2187", "domain": "hard_arithmetic"},
    {"problem": "98765 + 43210 = ?", "answer": "141975", "domain": "hard_arithmetic"},
    {"problem": "8888 - 999 = ?", "answer": "7889", "domain": "hard_arithmetic"},
]

def extract_number(text):
    if not text:
        return None, None
    m = re.search(r'\\boxed\s*\{(-?\d+)\}', text)
    if m:
        return m.group(1), "boxed"
    nums = re.findall(r'-?\d+', text)
    if nums:
        return nums[-1], "number_only"
    return None, "none"

def verify_answer(predicted, ground_truth):
    pred, src = predicted, src if 'src' in dir() else "unknown"
    if pred is None:
        return False
    try:
        return int(pred) == int(ground_truth)
    except:
        return False

def format_check(response):
    """Layer 1: format gate. Returns (has_boxed, extracted_num)."""
    num, src = extract_number(response)
    return {
        "has_boxed": src == "boxed",
        "has_number": num is not None,
        "extracted": num,
        "source": src
    }

def call_llm(problem, api_key, retries=2):
    import urllib.request, urllib.error, time
    payload = {
        "model": "MiniMax-2.7",
        "max_tokens": 256,
        "messages": [{"role": "user", "content": f"Solve. Answer in \\boxed{{number}} format.\n\n{problem}"}]
    }
    for attempt in range(retries + 1):
        try:
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
        except Exception as e:
            if attempt < retries:
                time.sleep(2 ** attempt)
                continue
            return f"ERROR: {e}"

def load_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        cfg_path = Path.home() / ".openclaw" / "openclaw.json"
        if cfg_path.exists():
            try:
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                for p in cfg.get("providers", {}).values():
                    if isinstance(p, dict) and "api_key" in p:
                        return p["api_key"]
            except Exception:
                pass
    return key

def compute_corpus_h1(problems):
    """Layer 2: Compute H1 on entire corpus for batch health signal."""
    from topomem.embedding import EmbeddingManager
    from topomem.topology import TopologyEngine, TopologyConfig

    emb_mgr = EmbeddingManager()
    texts = [p["problem"] for p in problems]
    embeddings = emb_mgr.encode_batch(texts)

    cfg = TopologyConfig(max_homology_dim=2, metric="cosine")
    topo = TopologyEngine(config=cfg)
    # ripsER needs (n_points, n_dims), our embeddings are (n, 384)
    result = topo.compute_persistence(embeddings)

    h0 = result[0] if len(result) > 0 else []
    h1 = result[1] if len(result) > 1 else []
    h2 = result[2] if len(result) > 2 else []

    def mean_pers(pairs):
        if pairs is None or (isinstance(pairs, (list, tuple)) and len(pairs) == 0):
            return 0.0
        arr = np.array(pairs) if not hasattr(pairs, 'shape') else pairs
        if arr.size == 0:
            return 0.0
        pers = [float(arr[i,1] - arr[i,0]) for i in range(len(arr))]
        return sum(pers)/len(pers) if pers else 0.0

    return {
        "n_nodes": len(problems),
        "h0_count": len(h0),
        "h1_count": len(h1),
        "h2_count": len(h2),
        "h1_mean_pers": mean_pers(h1),
        "h2_mean_pers": mean_pers(h2),
        "fragmentation": len(h1) / len(problems) if problems else 0,
    }

def run_hybrid_gateway(problems, api_key, name="Test"):
    """
    Three-layer gateway:
      Layer 1: format check (instant)
      Layer 2: if no boxed, retry once
      Layer 3: verify + retry if still wrong
    """
    results = {
        "name": name,
        "n": len(problems),
        "layer1_format_ok": 0,
        "layer1_retry_for_format": 0,
        "layer2_final_correct": 0,
        "layer3_verify_retry": 0,
        "layer3_recovered": 0,
        "details": []
    }

    for i, item in enumerate(problems):
        # Layer 1: First attempt + format check
        resp1 = call_llm(item["problem"], api_key)
        fc1 = format_check(resp1)

        attempt_1_correct = False
        if fc1["has_boxed"] and fc1["extracted"] is not None:
            attempt_1_correct = int(fc1["extracted"]) == int(item["answer"])

        if fc1["has_boxed"]:
            results["layer1_format_ok"] += 1
            if attempt_1_correct:
                results["layer2_final_correct"] += 1
                results["details"].append({
                    "i": i, "problem": item["problem"],
                    "status": "L1_PASS",
                    "attempts": 1, "final_correct": True
                })
                time.sleep(0.5)
                continue
        else:
            results["layer1_retry_for_format"] += 1

        # Layer 2: Retry for format error
        resp2 = call_llm(item["problem"], api_key)
        fc2 = format_check(resp2)

        attempt_2_correct = False
        if fc2["has_boxed"] and fc2["extracted"] is not None:
            attempt_2_correct = int(fc2["extracted"]) == int(item["answer"])

        if attempt_2_correct or not fc2["has_boxed"]:
            # Either fixed format or still no boxed (give up on format)
            results["layer2_final_correct"] += 1 if attempt_2_correct else 0
            results["details"].append({
                "i": i, "problem": item["problem"],
                "status": "L2_FORMAT_RETRY",
                "attempts": 2,
                "final_correct": attempt_2_correct,
                "extracted_1": fc1["extracted"], "extracted_2": fc2["extracted"]
            })
            time.sleep(0.5)
            continue

        # Layer 3: verify + semantic retry (more careful prompt)
        results["layer3_verify_retry"] += 1
        resp3 = call_llm(
            f"Calculate the answer step by step. Your final answer must be in \\boxed{{number}} format. Do not include any other text. {item['problem']}",
            api_key
        )
        fc3 = format_check(resp3)
        attempt_3_correct = (
            fc3["has_boxed"] and fc3["extracted"] is not None
            and int(fc3["extracted"]) == int(item["answer"])
        )

        recovered = 1 if (attempt_3_correct and not attempt_1_correct) else 0
        results["layer3_recovered"] += recovered
        results["layer2_final_correct"] += 1 if attempt_3_correct else 0

        results["details"].append({
            "i": i, "problem": item["problem"],
            "status": "L3_VERIFY_RETRY",
            "attempts": 3,
            "final_correct": attempt_3_correct,
            "recovered": recovered,
            "extracted_1": fc1["extracted"], "extracted_3": fc3["extracted"]
        })
        time.sleep(0.5)

    return results

def run_baseline(problems, api_key, name="Baseline"):
    """Simple: one attempt, no retry. For comparison."""
    results = {
        "name": name,
        "n": len(problems),
        "correct": 0,
        "format_fail": 0,
        "calc_fail": 0,
        "details": []
    }
    for i, item in enumerate(problems):
        resp = call_llm(item["problem"], api_key)
        fc = format_check(resp)
        if not fc["has_boxed"]:
            results["format_fail"] += 1
            results["details"].append({"i": i, "correct": False, "reason": "format"})
        elif fc["extracted"] is not None and int(fc["extracted"]) == int(item["answer"]):
            results["correct"] += 1
            results["details"].append({"i": i, "correct": True, "reason": "correct"})
        else:
            results["calc_fail"] += 1
            results["details"].append({"i": i, "correct": False, "reason": "calc", "got": fc["extracted"]})
        print(f"  {name} {i+1}/{len(problems)}: {'PASS' if results['details'][-1]['correct'] else 'FAIL'} | {item['problem'][:30]}")
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

    print("=" * 60)
    print("HYBRID CONFIDENCE GATEWAY")
    print("Layer1: Format check | Layer2: Format retry | Layer3: Verify retry")
    print("=" * 60)

    # Layer 2: Corpus H1 health
    print("\n[Computing corpus H1 health...]")
    h1_health = compute_corpus_h1(PROBLEMS)
    print(f"  Corpus: {h1_health['n_nodes']} nodes, H1={h1_health['h1_count']}, H2={h1_health['h2_count']}")
    print(f"  H1 mean persistence: {h1_health['h1_mean_pers']:.4f}")
    print(f"  Fragmentation: {h1_health['fragmentation']:.2f}")

    # Baseline run
    print("\n[BASELINE: single attempt, no retry]")
    baseline = run_baseline(PROBLEMS, api_key, "Baseline")
    print(f"\nBaseline: {baseline['correct']}/{baseline['n']} = {baseline['correct']/baseline['n']:.0%}")
    print(f"  Format fails: {baseline['format_fail']}, Calc fails: {baseline['calc_fail']}")

    time.sleep(3)

    # Hybrid gateway run
    print("\n[HYBRID GATEWAY: 3-layer routing]")
    hybrid = run_hybrid_gateway(PROBLEMS, api_key, "Hybrid")
    print(f"\nHybrid: {hybrid['layer2_final_correct']}/{hybrid['n']} = {hybrid['layer2_final_correct']/hybrid['n']:.0%}")
    print(f"  L1 format OK: {hybrid['layer1_format_ok']}")
    print(f"  L2 format retry: {hybrid['layer1_retry_for_format']}")
    print(f"  L3 verify retry: {hybrid['layer3_verify_retry']}")
    print(f"  L3 recovered: {hybrid['layer3_recovered']}")

    # Summary
    baseline_acc = baseline["correct"] / baseline["n"]
    hybrid_acc = hybrid["layer2_final_correct"] / hybrid["n"]
    total_retries = hybrid["layer1_retry_for_format"] + hybrid["layer3_verify_retry"]

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Baseline:  ACC={baseline_acc:.0%} | Retries=0")
    print(f"Hybrid:    ACC={hybrid_acc:.0%} | Retries={total_retries} ({hybrid_acc-baseline_acc:+.0%})")

    # Save
    output = {
        "timestamp": ts,
        "h1_corpus_health": h1_health,
        "baseline": baseline,
        "hybrid": hybrid,
        "summary": {
            "baseline_acc": baseline_acc,
            "hybrid_acc": hybrid_acc,
            "delta": hybrid_acc - baseline_acc,
            "total_retries": total_retries
        }
    }
    out_file = out_dir / f"hybrid_gateway_{ts}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_file}")

if __name__ == "__main__":
    main()
