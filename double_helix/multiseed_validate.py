"""Multi-seed validation of the inverted-U hypothesis.

Runs boundary_scan with 3 seeds, collects results, computes statistics.
Only tests: single_shot / blind_retry / feedback_retry on trivial difficulty.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.chdir(Path(__file__).resolve().parents[1])

if os.name == "nt":
    try:
        import winreg
        _k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")
        _mp, _ = winreg.QueryValueEx(_k, "Path")
        _k2 = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Environment")
        _up, _ = winreg.QueryValueEx(_k2, "Path")
        os.environ["PATH"] = _mp + ";" + _up
    except Exception:
        pass

from double_helix.boundary_scan import run_scan, print_boundary_map
from core.capability_benchmark import generate_code_tasks


def run_multi_seed(
    solver_type: str,
    seeds: list,
    num_tasks: int = 20,
    budget: int = 2,
    difficulty: str = "trivial",
    gguf_path: str = "",
    model=None,
    tokenizer=None,
):
    import time
    import hashlib
    from collections import defaultdict
    
    all_cells = []
    
    # Generate unique experiment signature to avoid mixing results from different parameters
    param_str = f"{solver_type}_{difficulty}_b{budget}_t{num_tasks}"
    if gguf_path:
        # Use model file base name + short hash to keep filename readable
        model_name = Path(gguf_path).stem
        short_hash = hashlib.md5(gguf_path.encode()).hexdigest()[:6]
        param_str += f"_{model_name}_{short_hash}"
    
    results_file = Path(__file__).resolve().parents[1] / "double_helix" / "results" / f"multiseed_{param_str}_inprogress.json"
    
    # Load existing results if interrupted
    if results_file.exists():
        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            # Verify parameters match to prevent loading wrong results
            if existing_data.get('parameters', {}):
                saved_params = existing_data['parameters']
                if (saved_params.get('solver_type') != solver_type or
                    saved_params.get('difficulty') != difficulty or
                    saved_params.get('budget') != budget or
                    saved_params.get('num_tasks') != num_tasks or
                    saved_params.get('gguf_path') != gguf_path):
                    print("\n⚠️  WARNING: Existing results file has different parameters!")
                    print("  Starting fresh experiment instead of resuming.")
                    all_cells = []
                else:
                    all_cells = existing_data.get('cells', [])
                    completed_seeds = set(str(c.get('seed')) for c in all_cells if c.get('seed') is not None)
                    seeds = [s for s in seeds if str(s) not in completed_seeds]
                    if seeds:
                        print(f"\n{'='*80}")
                        print(f"  RESUMING EXPERIMENT")
                        print(f"  {len(completed_seeds)} seeds completed, {len(seeds)} seeds remaining")
                        print(f"{'='*80}")
                    else:
                        print("  All seeds completed, loading final results")
                        return all_cells
        except Exception as e:
            print(f"  Failed to load existing results: {e}")
    
    # Save parameters with results
    experiment_metadata = {
        "parameters": {
            "solver_type": solver_type,
            "difficulty": difficulty,
            "budget": budget,
            "num_tasks": num_tasks,
            "gguf_path": gguf_path,
            "started_at": datetime.now().isoformat()
        },
        "cells": all_cells
    }
    
    print(f"\n{'='*80}")
    print(f"  MULTI-SEED VALIDATION")
    print(f"  Solver: {solver_type}")
    print(f"  Seeds: {seeds}")
    print(f"  Tasks per seed: {num_tasks}")
    print(f"  Difficulty: {difficulty}")
    print(f"  Budget: {budget}")
    if gguf_path:
        print(f"  Model: {Path(gguf_path).name}")
    print(f"  Results file: {results_file.name}")
    print(f"{'='*80}")
    
    start_time = time.time()
    task_stats = defaultdict(lambda: {"total": 0, "single": 0, "blind": 0, "feedback": 0})
    
    for seed_idx, seed in enumerate(seeds, 1):
        print(f"\n{'='*80}")
        print(f"  SEED {seed} ({seed_idx}/{len(seeds)})")
        print(f"  Start time: {time.strftime('%H:%M:%S')}")
        print(f"{'='*80}")
        
        seed_start = time.time()
        
        try:
            result = run_scan(
                solver_type=solver_type,
                variants=["standard"],
                conditions=["single_shot", "blind_retry", "feedback_retry"],
                budgets=[budget],
                num_tasks=num_tasks,
                seed=seed,
                model=model,
                tokenizer=tokenizer,
                gguf_path=gguf_path,
                difficulty=difficulty,
            )

            for cell in result["cells"]:
                cell["seed"] = seed
                all_cells.append(cell)
                
                # Update task stats
                cond = cell["condition"]
                # Map condition to stats key
                cond_key = {
                    "single_shot": "single",
                    "blind_retry": "blind",
                    "feedback_retry": "feedback"
                }.get(cond, cond)
                for task in cell.get("per_task_results", []):
                    bt = task.get("bug_type", "unknown")
                    task_stats[bt]["total"] += 1
                    if task.get("solved"):
                        task_stats[bt][cond_key] += 1

            # Print seed-level results
            print(f"\n  SEED {seed} RESULTS:")
            print(f"  {'-'*60}")
            for cell in result["cells"]:
                cond = cell["condition"]
                rate = cell.get("rate", 0)
                print(f"  {cond:20s}: {cell.get('solved', 0)}/{cell.get('total', 0)} = {rate:.1%}")
            
            # Calculate delta
            blind_cell = next((c for c in result["cells"] if c["condition"] == "blind_retry"), None)
            fb_cell = next((c for c in result["cells"] if c["condition"] == "feedback_retry"), None)
            if blind_cell and fb_cell:
                delta = fb_cell.get("rate", 0) - blind_cell.get("rate", 0)
                print(f"  {'-'*60}")
                print(f"  Delta(Feedback - Blind): {delta:+.1%}")
            
            # Save intermediate results
            intermediate_data = experiment_metadata.copy()
            intermediate_data['cells'] = all_cells
            intermediate_data['metadata'] = {
                'completed_seeds': [str(s) for s in seeds[:seed_idx]],
                'remaining_seeds': [str(s) for s in seeds[seed_idx:]],
                'elapsed_time': time.time() - start_time,
                'seed_elapsed_time': time.time() - seed_start
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(intermediate_data, f, ensure_ascii=False, indent=2, default=str)
            
            print(f"  {'-'*60}")
            print(f"  Seed completed in {time.time() - seed_start:.1f} seconds")
            print(f"  Intermediate results saved")
            
            # Print task-level progress (top 5 bug types)
            if task_stats:
                print(f"\n  TOP 5 BUG TYPES BY FEEDBACK BENEFIT:")
                print(f"  {'-'*80}")
                print(f"  {'Bug Type':20s} | {'Total':>6} | {'Single':>6} | {'Blind':>6} | {'Feedback':>6} | {'Delta':>6}")
                print(f"  {'-'*80}")
                
                type_deltas = []
                for bt, stats in task_stats.items():
                    if stats["total"] >= 3:  # At least one per seed
                        blind_rate = stats["blind"] / stats["total"] if stats["total"] > 0 else 0
                        fb_rate = stats["feedback"] / stats["total"] if stats["total"] > 0 else 0
                        delta = fb_rate - blind_rate
                        type_deltas.append((delta, bt, stats))
                
                for delta, bt, stats in sorted(type_deltas, reverse=True)[:5]:
                    blind_rate = stats["blind"] / stats["total"] if stats["total"] > 0 else 0
                    fb_rate = stats["feedback"] / stats["total"] if stats["total"] > 0 else 0
                    print(f"  {bt[:20]:20s} | {stats['total']:6d} | {stats['single']:6d} | {stats['blind']:6d} | {stats['feedback']:6d} | {delta:+6.1%}")
        
        except Exception as e:
            print(f"  Error on seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            # Save partial results
            if all_cells:
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump({'cells': all_cells, 'error': str(e)}, f, ensure_ascii=False, indent=2, default=str)
            continue
    
    # Clean up in-progress file
    if results_file.exists():
        try:
            results_file.unlink()
        except Exception:
            pass
    
    print(f"\n{'='*80}")
    print(f"  EXPERIMENT COMPLETED")
    print(f"  Total elapsed time: {time.time() - start_time:.1f} seconds")
    print(f"{'='*80}")
    
    return all_cells


def compute_statistics(all_cells):
    from collections import defaultdict

    by_condition = defaultdict(list)
    for cell in all_cells:
        by_condition[cell["condition"]].append(cell["rate"])

    print(f"\n{'='*60}")
    print(f"  AGGREGATE RESULTS ACROSS SEEDS")
    print(f"{'='*60}")

    for cond in ["single_shot", "blind_retry", "feedback_retry"]:
        rates = by_condition[cond]
        n = len(rates)
        mean_rate = sum(rates) / n if n > 0 else 0
        std_rate = (sum((r - mean_rate) ** 2 for r in rates) / n) ** 0.5 if n > 1 else 0
        print(f"  {cond:20s}: {mean_rate:.1%} +/- {std_rate:.1%} (n={n})")

    blind_rates = by_condition["blind_retry"]
    fb_rates = by_condition["feedback_retry"]

    if blind_rates and fb_rates and len(blind_rates) == len(fb_rates):
        deltas = [f - b for f, b in zip(fb_rates, blind_rates)]
        mean_delta = sum(deltas) / len(deltas)
        positive_count = sum(1 for d in deltas if d > 0)
        print(f"\n  Delta(Feedback - Blind): {mean_delta:+.1%}")
        print(f"  Positive deltas: {positive_count}/{len(deltas)} seeds")

        if len(deltas) >= 3:
            try:
                from scipy.stats import mannwhitneyu, wilcoxon
                try:
                    stat, p = wilcoxon(fb_rates, blind_rates, alternative="greater")
                    print(f"  Wilcoxon signed-rank test: W={stat:.1f}, p={p:.4f}")
                except Exception:
                    stat, p = mannwhitneyu(fb_rates, blind_rates, alternative="greater")
                    print(f"  Mann-Whitney U test: U={stat:.1f}, p={p:.4f}")
                if p < 0.05:
                    print(f"  *** STATISTICALLY SIGNIFICANT (p < 0.05) ***")
                else:
                    print(f"  NOT statistically significant (p >= 0.05)")
            except ImportError:
                print(f"  (scipy not available, skipping significance test)")

    return by_condition


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--solver", choices=["search", "qwen", "phi4mini", "gemma4b"], default="phi4mini")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--tasks", type=int, default=20)
    parser.add_argument("--budget", type=int, default=2)
    parser.add_argument("--difficulty", default="trivial")
    args = parser.parse_args()

    model = None
    tokenizer = None
    gguf_path = ""

    if args.solver == "qwen":
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_name = "Qwen/Qwen2.5-0.5B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, local_files_only=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True, local_files_only=True)
        model.eval()
    elif args.solver == "phi4mini":
        gguf_path = str(Path(__file__).resolve().parents[1] / "double_helix" / "microsoft_Phi-4-mini-instruct-Q4_K_M.gguf")
    elif args.solver == "gemma4b":
        gguf_path = str(Path(__file__).resolve().parents[1] / "double_helix" / "google_gemma-3-4b-it-Q5_K_M (1).gguf")

    all_cells = run_multi_seed(
        solver_type=args.solver,
        seeds=args.seeds,
        num_tasks=args.tasks,
        budget=args.budget,
        difficulty=args.difficulty,
        gguf_path=gguf_path,
        model=model,
        tokenizer=tokenizer,
    )

    by_condition = compute_statistics(all_cells)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(__file__).resolve().parents[1] / "double_helix" / "results" / f"multiseed_{args.solver}_{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"solver": args.solver, "seeds": args.seeds, "difficulty": args.difficulty,
                    "budget": args.budget, "tasks_per_seed": args.tasks, "cells": all_cells}, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
