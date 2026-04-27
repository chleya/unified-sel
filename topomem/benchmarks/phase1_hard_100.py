"""
Phase 1: n=100 Harder MATH Test

Goal: baseline accuracy 50-70% (currently n=20 is near-ceiling at 80-100%)
Difficulty: 4 tiers (medium/hard/very hard/counterintuitive)
"""
import os, sys, json, time, re, random
from pathlib import Path

PROJECT_ROOT = Path(r"F:\unified-sel")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
import numpy as np

random.seed(42)

# Difficulty tiers - designed for 50-70% baseline
# tier1: medium (expected ~80%) - 2-digit × 2-digit, 3-digit + 3-digit
# tier2: hard (expected ~65%) - 3-digit × 3-digit, 4-digit mixed
# tier3: very hard (expected ~50%) - multi-step, 4-digit × 3-digit
# tier4: counterintuitive (expected ~35%) - large division, fractional reasoning

TIER1 = [
    ("27 × 38", 1026), ("54 × 29", 1566), ("63 × 47", 2961), ("81 × 56", 4536),
    ("45 × 67", 3015), ("72 × 38", 2736), ("39 × 84", 3276), ("58 × 43", 2494),
    ("234 + 567", 801), ("456 + 789", 1245), ("823 + 456", 1279), ("345 + 678", 1023),
    ("782 + 591", 1373), ("923 + 467", 1390), ("2345 - 1234", 1111), ("5678 - 2345", 3333),
    ("4567 - 1234", 3333), ("6789 - 3456", 3333), ("92 // 4", 23), ("87 // 3", 29),
    ("156 // 6", 26), ("207 // 9", 23), ("1008 // 7", 144), ("756 // 4", 189),
]

TIER2 = [
    ("234 × 56", 13104), ("345 × 78", 26910), ("456 × 67", 30552), ("567 × 89", 50463),
    ("1234 + 5678", 6912), ("2345 + 6789", 9134), ("3456 + 4321", 7777), ("4567 + 5432", 9999),
    ("7890 - 1234", 6656), ("8765 - 2345", 6420), ("9876 - 3456", 6420), ("6543 - 1234", 5309),
    ("1000 - 456", 544), ("2000 - 789", 1211), ("3000 - 1234", 1766), ("5000 - 2345", 2655),
    ("84 × 37", 3108), ("91 × 58", 5278), ("76 × 49", 3724), ("88 × 77", 6776),
    ("1024 // 8", 128), ("2025 // 9", 225), ("3025 // 5", 605), ("4096 // 16", 256),
    ("5625 // 25", 225), ("7056 // 28", 252), ("8649 // 93", 93), ("7744 // 88", 88),
]

TIER3 = [
    ("234 × 567", 132678), ("345 × 678", 233910), ("456 × 789", 359784), ("123 × 456", 56088),
    ("2345 + 6789", 9134), ("4567 + 8901", 13468), ("7890 + 1234", 9124), ("3456 + 7777", 11233),
    ("9876 - 5432", 4444), ("8765 - 4321", 4444), ("7654 - 3210", 4444), ("6543 - 2109", 4434),
    ("12345 - 6789", 5556), ("23456 - 7890", 15566), ("1111 × 3", 3333), ("2222 × 4", 8888),
    ("3333 × 3", 9999), ("4444 × 2", 8888), ("5555 × 2", 11110), ("6666 × 2", 13332),
    ("12345 // 5", 2469), ("23456 // 8", 2932), ("34567 // 7", 4938), ("45678 // 9", 5075),
    ("56789 // 11", 5162), ("67890 // 15", 4526), ("78901 // 13", 6069), ("89012 // 17", 5236),
]

TIER4 = [
    ("123456 ÷ 64", 1929), ("234567 ÷ 89", 2635), ("345678 ÷ 123", 2810),
    ("456789 ÷ 234", 1952), ("567890 ÷ 345", 1646), ("678901 ÷ 456", 1489),
    ("99 × 99", 9801), ("999 × 999", 998001), ("888 × 888", 788544), ("777 × 777", 603729),
    ("1111 × 9", 9999), ("11111 × 9", 99999), ("1234 × 99", 122166), ("2345 × 99", 232155),
    ("123456 + 78901", 202357), ("234567 + 89012", 323579), ("345678 + 90123", 435801),
    ("999999 - 123456", 876543), ("888888 - 234567", 654321), ("777777 - 345678", 432099),
]

# Build full problem set with tier labels
PROBLEMS = []
for p, a in TIER1:
    PROBLEMS.append({"problem": f"{p} = ?", "answer": str(a), "tier": 1})
for p, a in TIER2:
    PROBLEMS.append({"problem": f"{p} = ?", "answer": str(a), "tier": 2})
for p, a in TIER3:
    PROBLEMS.append({"problem": f"{p} = ?", "answer": str(a), "tier": 3})
for p, a in TIER4:
    PROBLEMS.append({"problem": f"{p} = ?", "answer": str(a), "tier": 4})

random.shuffle(PROBLEMS)
print(f"Total problems: {len(PROBLEMS)}")
for t in [1,2,3,4]:
    ct = sum(1 for p in PROBLEMS if p["tier"]==t)
    print(f"  Tier {t}: {ct} problems")

def extract_number(text):
    if not text: return None
    m = re.search(r'\\boxed\s*\{(-?\d+)\}', text)
    if m: return m.group(1)
    nums = re.findall(r'-?\d+', text)
    return nums[-1] if nums else None

def format_check(response):
    num = extract_number(response)
    m = re.search(r'\\boxed', response) if response else None
    return {"has_boxed": m is not None, "extracted": num}

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
            except: pass
    return key

def call_llm(problem, api_key, retries=3):
    import urllib.request, urllib.error
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

def run_test(problems, api_key, name, max_per_tier=None):
    results = {"name": name, "correct": 0, "format_fail": 0, "calc_fail": 0, "errors": 0, "tier_correct": {1:0,2:0,3:0,4:0}, "tier_total": {1:0,2:0,3:0,4:0}, "details": []}
    
    for item in problems:
        tier = item["tier"]
        if max_per_tier and results["tier_total"][tier] >= max_per_tier:
            continue
        results["tier_total"][tier] += 1
        
        resp = call_llm(item["problem"], api_key)
        fc = format_check(resp)
        
        if "ERROR" in str(resp):
            results["errors"] += 1
            results["details"].append({"i": len(results["details"]), "correct": False, "reason": "error", "problem": item["problem"][:30]})
            print(f"  {name} E/({results['tier_total'][tier]}) ERROR: {item['problem'][:25]}")
            time.sleep(2)
            continue
        
        if not fc["has_boxed"]:
            results["format_fail"] += 1
            results["details"].append({"i": len(results["details"]), "correct": False, "reason": "format", "problem": item["problem"][:30]})
            print(f"  {name} F/({results['tier_total'][tier]}) FORMAT: {item['problem'][:25]}")
        elif fc["extracted"] is None or int(fc["extracted"]) != int(item["answer"]):
            results["calc_fail"] += 1
            results["tier_correct"][tier] += 0
            results["details"].append({"i": len(results["details"]), "correct": False, "reason": "calc", "got": fc["extracted"], "problem": item["problem"][:30]})
            print(f"  {name} W/({results['tier_total'][tier]}) WRONG: {item['problem'][:25]} got={fc['extracted']} expected={item['answer']}")
        else:
            results["correct"] += 1
            results["tier_correct"][tier] += 1
            results["details"].append({"i": len(results["details"]), "correct": True, "problem": item["problem"][:30]})
            print(f"  {name} P/({results['tier_total'][tier]}) PASS: {item['problem'][:25]}")
        
        time.sleep(0.3)
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
    print("PHASE 1: n=100 Harder MATH Test")
    print("=" * 60)
    
    # Run full test (n=100, all tiers)
    print("\n[FULL TEST: 100 problems]")
    results = run_test(PROBLEMS, api_key, "Hard100")
    total = sum(results["tier_total"].values())
    print(f"\n{'='*40}")
    print(f"Hard100 Results: {results['correct']}/{total} = {results['correct']/total:.0%}")
    print(f"  Format fails: {results['format_fail']}")
    print(f"  Calc fails:   {results['calc_fail']}")
    print(f"  Errors:       {results['errors']}")
    print(f"\nBy tier:")
    for t in [1,2,3,4]:
        tot = results["tier_total"][t]
        cor = results["tier_correct"][t]
        if tot > 0:
            print(f"  Tier {t}: {cor}/{tot} = {cor/tot:.0%}")
    
    output_file = out_dir / f"phase1_hard100_{ts}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"timestamp": ts, "results": results, "summary": {
            "total_acc": results["correct"]/total if total else 0,
            "format_fail_rate": results["format_fail"]/total if total else 0,
            "calc_fail_rate": results["calc_fail"]/total if total else 0
        }}, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {output_file}")

if __name__ == "__main__":
    main()
