"""
topomem benchmarks - 性能基准测试系统

每次优化前先跑这个，记录结果，再做改动。
改动后再跑，对比数据再决定是否保留。

使用方式：
    python -m topomem.benchmarks.run_benchmarks

输出：
    topomem/benchmarks/results/YYYYMMDD_HHMMSS.json
"""

import time
import json
import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

# Hardcoded paths to avoid ambiguity when running as -m module
# Project root: F:\unified-sel
PROJECT_ROOT = Path(r"F:\unified-sel")
# TopoMem module dir: F:\unified-sel\topomem
TOPOMEM_ROOT = PROJECT_ROOT / "topomem"
RESULTS_DIR = TOPOMEM_ROOT / "benchmarks" / "results"
HF_CACHE = TOPOMEM_ROOT / "data" / "models" / "hf_cache"


def run_cmd(cmd, cwd=None, timeout=None):
    """运行命令并返回时间"""
    start = time.perf_counter()
    result = subprocess.run(
        cmd,
        shell=True,
        cwd=str(cwd or PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.perf_counter() - start
    return result, elapsed


def bench_full_suite():
    """完整测试套件时间"""
    print("  Running full test suite (this takes ~7min)...")
    result, elapsed = run_cmd(
        "python -m pytest tests/ -q --tb=no",
        cwd=TOPOMEM_ROOT,
        timeout=900,
    )
    return {
        "elapsed_s": round(elapsed, 1),
        "elapsed_formatted": f"{int(elapsed//60)}:{int(elapsed%60):02d}",
        "passed": result.stdout.count("passed"),
        "exit_code": result.returncode,
    }


def bench_integration_tests():
    """集成测试时间"""
    print("  Running integration tests...")
    result, elapsed = run_cmd(
        "python -m pytest tests/test_integration.py -q --tb=no",
        cwd=TOPOMEM_ROOT,
        timeout=600,
    )
    return {
        "elapsed_s": round(elapsed, 1),
        "elapsed_formatted": f"{int(elapsed//60)}:{int(elapsed%60):02d}",
        "passed": result.stdout.count("passed"),
        "exit_code": result.returncode,
    }


def bench_unit_tests():
    """单元测试时间（排除集成测试）"""
    print("  Running unit tests...")
    result, elapsed = run_cmd(
        "python -m pytest tests/ -q --tb=no --ignore=tests/test_integration.py",
        cwd=TOPOMEM_ROOT,
        timeout=300,
    )
    return {
        "elapsed_s": round(elapsed, 1),
        "passed": result.stdout.count("passed"),
        "exit_code": result.returncode,
    }


def bench_embedding_load():
    """Embedding 模型加载时间"""
    print("  Benchmarking embedding model load...")
    # Hardcode paths to avoid any Path resolution issues in subprocess
    proj_root = r"F:\unified-sel"
    topo_root = r"F:\unified-sel\topomem"
    hf_cache = r"F:\unified-sel\topomem\data\models\hf_cache"
    code = f"""
import sys, os, time
sys.path.insert(0, r'{proj_root}')
os.chdir(r'{topo_root}')
os.environ['HF_HOME'] = r'{hf_cache}'
os.environ['TRANSFORMERS_CACHE'] = r'{hf_cache}'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'{hf_cache}'
from topomem.embedding import EmbeddingManager
# 清除缓存
EmbeddingManager._global_model = None
t0 = time.perf_counter()
mgr = EmbeddingManager()
_ = mgr.model
load_time = time.perf_counter() - t0
print(f'LOAD_TIME:{{load_time:.3f}}')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=60,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "LOAD_TIME:" in line:
            load_time = float(line.split("LOAD_TIME:")[1].strip())
            return {"load_time_s": round(load_time, 3)}
    return {"load_time_s": None, "error": result.stdout[:200] + result.stderr[:200]}


def bench_tda():
    """TDA 计算时间"""
    print("  Benchmarking TDA...")
    proj_root = r"F:\unified-sel"
    topo_root = r"F:\unified-sel\topomem"
    hf_cache = r"F:\unified-sel\topomem\data\models\hf_cache"
    code = f"""
import sys, os, time, numpy as np
sys.path.insert(0, r'{proj_root}')
os.chdir(r'{topo_root}')
os.environ['HF_HOME'] = r'{hf_cache}'
os.environ['TRANSFORMERS_CACHE'] = r'{hf_cache}'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'{hf_cache}'
from topomem.topology import TopologyEngine
te = TopologyEngine()
results = {{}}
for n in [10, 50, 100, 200]:
    pts = np.random.rand(n, 2).astype(np.float64)
    t0 = time.perf_counter()
    r = te.compute_persistence(pts)
    t = (time.perf_counter() - t0) * 1000
    results[f'n{{n}}'] = round(t, 1)
print(f'TDA_RESULTS:{{results}}')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "TDA_RESULTS:" in line:
            import ast
            raw = line.split("TDA_RESULTS:")[1].strip()
            return {"tda_ms_by_size": ast.literal_eval(raw)}
    return {"error": (result.stdout + result.stderr)[:300]}


def bench_encoding():
    """Embedding 编码时间"""
    print("  Benchmarking encoding...")
    proj_root = r"F:\unified-sel"
    topo_root = r"F:\unified-sel\topomem"
    hf_cache = r"F:\unified-sel\topomem\data\models\hf_cache"
    code = f"""
import sys, os, time
sys.path.insert(0, r'{proj_root}')
os.chdir(r'{topo_root}')
os.environ['HF_HOME'] = r'{hf_cache}'
os.environ['TRANSFORMERS_CACHE'] = r'{hf_cache}'
os.environ['SENTENCE_TRANSFORMERS_HOME'] = r'{hf_cache}'
from topomem.embedding import EmbeddingManager
mgr = EmbeddingManager()
texts = ['hello world'] * 30
# warmup
_ = mgr.encode(texts[0])
# individual
t0 = time.perf_counter()
for t in texts:
    mgr.encode(t)
individual = (time.perf_counter() - t0) * 1000
# batch
t0 = time.perf_counter()
mgr.encode_batch(texts)
batch = (time.perf_counter() - t0) * 1000
results = {{
    'individual_30': round(individual, 1),
    'batch_30': round(batch, 1),
    'speedup_batch_vs_loop': round(individual / batch, 2)
}}
print(f'ENCODING_RESULTS:{{results}}')
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        encoding="utf-8",
        errors="replace",
    )
    output = result.stdout + result.stderr
    for line in output.split("\n"):
        if "ENCODING_RESULTS:" in line:
            import ast
            raw = line.split("ENCODING_RESULTS:")[1].strip()
            return {"encoding": ast.literal_eval(raw)}
    return {"error": (result.stdout + result.stderr)[:300]}


def run_all_benchmarks():
    """运行所有基准测试"""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "timestamp": datetime.now().isoformat(),
        "session_id": ts,
        "tests": {},
    }

    benchmarks = [
        ("full_suite", bench_full_suite),
        ("integration_tests", bench_integration_tests),
        ("unit_tests", bench_unit_tests),
        ("embedding_load", bench_embedding_load),
        ("tda", bench_tda),
        ("encoding", bench_encoding),
    ]

    for name, fn in benchmarks:
        print(f"\n[{name}]")
        try:
            report["tests"][name] = fn()
            print(f"  -> {report['tests'][name]}")
        except Exception as e:
            import traceback
            report["tests"][name] = {"error": str(e), "traceback": traceback.format_exc()[:200]}
            print(f"  -> ERROR: {e}")

    # 保存结果
    result_file = RESULTS_DIR / f"{ts}.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {result_file}")

    # 和上次结果对比
    compare_with_previous(result_file, report)

    return report


def compare_with_previous(current_file, current_report):
    """和上次基准对比"""
    result_files = sorted(RESULTS_DIR.glob("*.json"))
    if len(result_files) < 2:
        print("  (首次基准，暂无历史数据对比)")
        return

    prev_file = result_files[-2]
    with open(prev_file) as f:
        prev = json.load(f)

    print(f"\n{'='*60}")
    print(f"对比: {prev_file.name} vs {current_file.name}")
    print(f"{'='*60}")

    def get_primary(v, key):
        if isinstance(v, dict) and key in v:
            return v[key]
        return None

    for key in prev.get("tests", {}):
        if key not in current_report["tests"]:
            continue
        p = prev["tests"][key]
        c = current_report["tests"][key]
        if "error" in p and "error" not in c:
            print(f"  {key}: ERROR -> FIXED")
        elif "error" in c and "error" not in p:
            print(f"  {key}: OK -> BROKEN")
        elif "error" in p and "error" in c:
            continue

        time_keys = ["elapsed_s", "load_time_s"]
        for tk in time_keys:
            pv = get_primary(p, tk)
            cv = get_primary(c, tk)
            if pv and cv:
                diff = cv - pv
                pct = (diff / pv) * 100
                symbol = "📈变慢" if diff > 0.01 else "📉变快" if diff < -0.01 else "➖相同"
                print(
                    f"  {key:25s}: {pv:8.2f}s → {cv:8.2f}s  "
                    f"({diff:+7.2f}s, {pct:+6.1f}%) {symbol}"
                )


if __name__ == "__main__":
    print(f"TopoMem 基准测试系统")
    print(f"=" * 40)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"TopoMem 目录: {TOPOMEM_ROOT}")
    print(f"结果保存目录: {RESULTS_DIR}")
    print()

    report = run_all_benchmarks()

    print("\n" + "=" * 60)
    print("基准测试完成!")
    print("=" * 60)
