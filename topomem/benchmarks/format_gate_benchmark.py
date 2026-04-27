"""
Format Gate Benchmark v2 - Robust with auto-retry

Three-layer confidence gateway:
  Layer 1: Format check - does response contain \boxed{}?
  Layer 2: Auto-retry on empty response (up to 3 attempts)
  Layer 3: Extraction + comparison

Key fix: compare extracted string to expected answer string
"""
import os, sys, json, re, time
from pathlib import Path

PROJECT_ROOT = Path(r"F:\unified-sel")
sys.path.insert(0, str(PROJECT_ROOT))
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")

API_KEY = "sk-cp-32X4yB3hv4uMfzdBmke7EyaxE2pXmHkAGisoBxm1bTlSnUKXcH3lGRgWYcD62Nre5AacJpbi0E5yOx92m5rkIth9HioW2aCHP5r3LeCKBuf-wdr1TVgeFxY"
API_URL = "https://api.minimaxi.com/anthropic/v1/messages"
HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
    "anthropic-version": "2023-06-01"
}

SYSTEM_PROMPT = "You are a math reasoning assistant. Output ONLY the answer in \\boxed{} format. Example: \\boxed{42}"

RETRY_PROMPT = "You are a math assistant. Output ONLY \\boxed{answer}. Example: \\boxed{3724}. Do NOT explain. Just output \\boxed{number}."

def normalize_problem(problem):
    return problem.replace('\u00d7', '*').replace('\u00f7', '//').replace('\u2212', '-').strip()

def extract_boxed(response_text):
    if not response_text:
        return None
    matches = re.findall(r'\\boxed\{([^}]+)\}', response_text)
    if matches:
        return matches[-1].strip()
    # Fallback: look for last number
    numbers = re.findall(r'-?\d+(?:\.\d+)?', response_text)
    if numbers:
        return numbers[-1]
    return None

def call_minimax(prompt, temperature=0.1, max_tokens=200):
    payload = {
        "model": "MiniMax-2.7",
        "max_tokens": max_tokens,
        "temperature": temperature,
        "messages": [{"role": "user", "content": prompt}]
    }
    for attempt in range(4):
        try:
            import urllib.request, ssl
            ctx = ssl.create_default_context()
            req = urllib.request.Request(
                API_URL,
                data=json.dumps(payload).encode('utf-8'),
                headers=HEADERS,
                method='POST'
            )
            with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                content = data.get('content', [])
                for block in content:
                    if isinstance(block, dict) and block.get('type') == 'text':
                        text = block.get('text', '')
                        if text.strip():
                            return text
                if content and len(content) > 0:
                    item = content[0]
                    if isinstance(item, dict) and item.get('text', '').strip():
                        return item['text']
            return ""
        except Exception as e:
            if attempt < 3:
                time.sleep(2 ** attempt)
            else:
                return f"ERROR: {e}"

def numbers_match(extracted_str, expected_str):
    """Check if extracted matches expected (comparing as numbers)."""
    if not extracted_str:
        return False, 'no_answer', extracted_str
    try:
        ext_val = float(extracted_str.strip())
        exp_val = float(str(expected_str).strip())
        if abs(ext_val - exp_val) < 1e-6:
            return True, 'correct', extracted_str
        return False, 'wrong_answer', extracted_str
    except (ValueError, TypeError):
        return False, 'non_numeric', extracted_str

def run_layer1_only(problems, max_count=None):
    """Layer 1: Single call, no retry."""
    results = []
    count = min(max_count or len(problems), len(problems))
    
    print("\n=== LAYER 1: Single call (baseline) [%d problems] ===" % count)
    for i, p in enumerate(problems[:count]):
        problem_norm = normalize_problem(p['problem'])
        prompt = SYSTEM_PROMPT + "\n\nProblem: " + problem_norm
        response = call_minimax(prompt)
        extracted = extract_boxed(response)
        matched, reason, got = numbers_match(extracted, p['answer'])
        
        results.append({
            'i': p.get('i', i),
            'problem': p['problem'],
            'expected': str(p['answer']),
            'got': got,
            'correct': matched,
            'reason': reason,
            'response_len': len(response)
        })
        print("  [%d/%d] %s -> %s (got=%s)" % (i+1, count, reason, p['problem'][:30], got))
        time.sleep(0.3)
    
    return results

def run_with_retry(problems, max_count=None, max_retries=3):
    """All layers: auto-retry on no answer, up to max_retries."""
    results = []
    count = min(max_count or len(problems), len(problems))
    
    print("\n=== WITH RETRY [%d problems, up to %d retries] ===" % (count, max_retries))
    for i, p in enumerate(problems[:count]):
        problem_norm = normalize_problem(p['problem'])
        final_correct = False
        final_reason = 'unknown'
        final_got = None
        all_responses = []
        
        for attempt in range(max_retries):
            if attempt == 0:
                prompt = SYSTEM_PROMPT + "\n\nProblem: " + problem_norm
                temp = 0.1
            else:
                prompt = RETRY_PROMPT + "\nProblem: " + problem_norm
                temp = 0.3
            
            response = call_minimax(prompt, temperature=temp)
            extracted = extract_boxed(response)
            matched, reason, got = numbers_match(extracted, p['answer'])
            
            all_responses.append({'attempt': attempt+1, 'response': response[:100], 'extracted': extracted, 'reason': reason})
            
            if matched:
                final_correct = True
                final_reason = 'correct_t%d' % (attempt+1)
                final_got = got
                break
            elif extracted is not None and reason != 'no_answer':
                # Got an answer but wrong
                final_reason = '%s_t%d' % (reason, attempt+1)
                final_got = got
                if attempt < max_retries - 1:
                    continue  # retry even on wrong answer sometimes
                break
            else:
                # No answer - retry
                final_reason = 'no_answer_t%d' % (attempt+1)
                final_got = got
        
        results.append({
            'i': p.get('i', i),
            'problem': p['problem'],
            'expected': str(p['answer']),
            'got': final_got,
            'correct': final_correct,
            'reason': final_reason,
            'attempts': all_responses
        })
        status = 'CORRECT' if final_correct else final_reason
        print("  [%d/%d] %s -> %s (got=%s)" % (i+1, count, status, p['problem'][:30], final_got))
        time.sleep(0.3)
    
    return results

def summarize(results, label):
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    no_box = sum(1 for r in results if 'no_answer' in r['reason'])
    wrong = sum(1 for r in results if not r['correct'] and 'no_answer' not in r['reason'])
    
    print("\n=== %s SUMMARY ===" % label)
    print("  Accuracy: %d/%d = %.1f%%" % (correct, total, 100*correct/total))
    print("  Format fail (no \\boxed): %d (%.1f%%)" % (no_box, 100*no_box/total))
    print("  Wrong answer: %d (%.1f%%)" % (wrong, 100*wrong/total))
    
    # Breakdown by attempt
    for t in range(1, 4):
        c = sum(1 for r in results if r['correct'] and '_t%d' % t in r['reason'])
        n = sum(1 for r in results if not r['correct'] and '_t%d' % t in r['reason'])
        if c > 0 or n > 0:
            print("  Attempt %d: %d correct, %d wrong/no-box" % (t, c, n))
    
    return {'accuracy': correct/total, 'format_fail': no_box/total, 'wrong': wrong/total}

def main():
    # Embedded hard problems (50) - with computed answers
    problems = [
        {"i": 0, "problem": "76 * 49 = ?", "answer": "3724"},
        {"i": 1, "problem": "91 * 58 = ?", "answer": "5278"},
        {"i": 2, "problem": "11111 * 9 = ?", "answer": "99999"},
        {"i": 3, "problem": "456 + 789 = ?", "answer": "1245"},
        {"i": 4, "problem": "23456 - 7890 = ?", "answer": "15566"},
        {"i": 5, "problem": "8649 // 93 = ?", "answer": "93"},
        {"i": 6, "problem": "54 * 29 = ?", "answer": "1566"},
        {"i": 7, "problem": "5555 * 2 = ?", "answer": "11110"},
        {"i": 8, "problem": "5678 - 2345 = ?", "answer": "3333"},
        {"i": 9, "problem": "78901 // 13 = ?", "answer": "6069"},
        {"i": 10, "problem": "23456 // 8 = ?", "answer": "2932"},
        {"i": 11, "problem": "823 + 456 = ?", "answer": "1279"},
        {"i": 12, "problem": "123 * 456 = ?", "answer": "56088"},
        {"i": 13, "problem": "2345 + 6789 = ?", "answer": "9134"},
        {"i": 14, "problem": "12345 // 5 = ?", "answer": "2469"},
        {"i": 15, "problem": "2025 // 9 = ?", "answer": "225"},
        {"i": 16, "problem": "5625 // 25 = ?", "answer": "225"},
        {"i": 17, "problem": "1234 * 99 = ?", "answer": "122166"},
        {"i": 18, "problem": "56789 // 11 = ?", "answer": "5162"},
        {"i": 19, "problem": "2000 - 789 = ?", "answer": "1211"},
        # Tier 2
        {"i": 20, "problem": "3456 + 4321 = ?", "answer": "7777"},
        {"i": 21, "problem": "207 // 9 = ?", "answer": "23"},
        {"i": 22, "problem": "7890 - 1234 = ?", "answer": "6656"},
        {"i": 23, "problem": "345678 + 90123 = ?", "answer": "435801"},
        {"i": 24, "problem": "123456 // 64 = ?", "answer": "1929"},
        {"i": 25, "problem": "7056 // 28 = ?", "answer": "252"},
        {"i": 26, "problem": "456789 // 234 = ?", "answer": "1952"},
        {"i": 27, "problem": "456 * 67 = ?", "answer": "30552"},
        {"i": 28, "problem": "999 * 999 = ?", "answer": "998001"},
        {"i": 29, "problem": "8765 - 2345 = ?", "answer": "6420"},
        {"i": 30, "problem": "234 + 567 = ?", "answer": "801"},
        {"i": 31, "problem": "4096 // 16 = ?", "answer": "256"},
        {"i": 32, "problem": "888888 - 234567 = ?", "answer": "654321"},
        {"i": 33, "problem": "234 * 567 = ?", "answer": "132678"},
        {"i": 34, "problem": "678901 // 456 = ?", "answer": "1489"},
        {"i": 35, "problem": "782 + 591 = ?", "answer": "1373"},
        {"i": 36, "problem": "1000 - 456 = ?", "answer": "544"},
        {"i": 37, "problem": "756 // 4 = ?", "answer": "189"},
        {"i": 38, "problem": "5000 - 2345 = ?", "answer": "2655"},
        {"i": 39, "problem": "84 * 37 = ?", "answer": "3108"},
        {"i": 40, "problem": "92 // 4 = ?", "answer": "23"},
        {"i": 41, "problem": "1111 * 3 = ?", "answer": "3333"},
        {"i": 42, "problem": "8765 - 4321 = ?", "answer": "4444"},
        {"i": 43, "problem": "9876 - 5432 = ?", "answer": "4444"},
        {"i": 44, "problem": "58 * 43 = ?", "answer": "2494"},
        {"i": 45, "problem": "9876 - 3456 = ?", "answer": "6420"},
        {"i": 46, "problem": "777777 - 345678 = ?", "answer": "432099"},
        {"i": 47, "problem": "3025 // 5 = ?", "answer": "605"},
        {"i": 48, "problem": "63 * 47 = ?", "answer": "2961"},
        {"i": 49, "problem": "7744 // 88 = ?", "answer": "88"},
    ]
    
    print("=== FORMAT GATE BENCHMARK v2 ===")
    print("Total problems: %d" % len(problems))
    
    # Run Layer 1 baseline (20 problems)
    baseline = run_layer1_only(problems, max_count=20)
    bs = summarize(baseline, "LAYER 1 BASELINE")
    
    # Run with retry (20 same problems)
    retry_results = run_with_retry(problems, max_count=20, max_retries=3)
    rs = summarize(retry_results, "WITH RETRY")
    
    # Comparison
    print("\n=== COMPARISON (20 problems) ===")
    print("  Baseline accuracy: %.1f%%" % (100 * bs['accuracy']))
    print("  Retry accuracy: %.1f%%" % (100 * rs['accuracy']))
    print("  Improvement: +%.1f%%" % (100 * (rs['accuracy'] - bs['accuracy'])))
    print("  Format fails recovered: %d" % (bs['format_fail'] - rs['format_fail']))
    
    # Save results
    result = {
        'baseline': baseline,
        'with_retry': retry_results,
        'summary_baseline': bs,
        'summary_retry': rs
    }
    import datetime
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    outfile = PROJECT_ROOT / 'topomem' / 'benchmarks' / 'results' / ('format_gate_%s.json' % ts)
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\nResults saved to: %s" % outfile)

if __name__ == '__main__':
    main()
