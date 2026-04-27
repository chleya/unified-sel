"""
实验 9：Hub Neuron 490 消融实验

目标：验证 hub neuron #490 的因果特异性重要性。

设计：
  - Condition 1 (baseline): 完整模型
  - Condition 2 (hub_490): 将 hidden_dim[490] 在所有 MLP 层置零
  - Condition 3 (high_pr_control): 将 L23 高 PageRank neuron #159 置零（对照）
  - Condition 4 (random_control): 将随机 neuron #100 置零（基线噪声）

判断标准：
  - Go:  #490 ablation 影响 > high_pr_control ablation 影响
  - Hold: #490 有影响，但不比特异性
  - No-Go: #490 与对照无显著差别

预计耗时：~3 小时（3 次模型加载 + MMLU 评估）
"""

from __future__ import annotations

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def _write_checkpoint(cond_name, neuron_indices, ppl, subject_results):
    """Write per-subject streaming checkpoint — merges with existing checkpoint data."""
    output_dir = Path("results/weight_graph/exp09")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "neuron_ablation_checkpoint.json"

    def cat_mean(cat_set):
        accs = [subject_results[s]["accuracy"]
                for s in cat_set if s in subject_results]
        return float(np.mean(accs)) if accs else 0.0

    # Load existing checkpoint to preserve other conditions
    existing = {}
    if checkpoint_path.exists():
        try:
            with open(checkpoint_path) as f:
                existing = json.load(f)
        except (json.JSONDecodeError, IOError):
            existing = {}

    # Merge: existing conditions + this condition's current state
    existing.setdefault("conditions", {})
    existing["conditions"][cond_name] = {
        "neurons_ablated": neuron_indices,
        "perplexity": ppl,
        "per_subject": subject_results,
        "category_means": {
            "STEM": cat_mean(STEM_SET),
            "Humanities": cat_mean(HUMANITIES_SET),
            "Social": cat_mean(SOCIAL_SET),
        },
        "overall_mean_acc": float(np.mean([subject_results[s]["accuracy"] for s in subject_results])) if subject_results else 0.0,
    }

    with open(checkpoint_path, "w") as f:
        json.dump(existing, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


# ============================================================
# Neuron ablation utilities
# ============================================================

def zero_neuron_all_layers(model, neuron_idx: int):
    """
    将所有 MLP 层中的指定 hidden dimension 置零。

    对每个 SwiGLU MLP 层：
      - gate_proj.weight[:, neuron_idx] = 0   (该维度的输入门权重)
      - up_proj.weight[:, neuron_idx] = 0      (该维度的上升权重)
      - down_proj.weight[neuron_idx, :] = 0    (该维度的输出权重)
    """
    count = 0
    for name, param in model.named_parameters():
        if "gate_proj.weight" in name or "up_proj.weight" in name:
            param.data[:, neuron_idx] = 0.0
            count += 1
        elif "down_proj.weight" in name:
            param.data[neuron_idx, :] = 0.0
            count += 1
    return count


def reset_neurons(model, neuron_indices: List[int]):
    """将多个 neuron 置零（一次性操作，更高效）。"""
    for ni in neuron_indices:
        zero_neuron_all_layers(model, ni)


# ============================================================
# Evaluation
# ============================================================

def evaluate_subject(model, tokenizer, subject: str) -> Dict:
    """评估单个 MMLU subject，返回 accuracy 和样本数。"""
    try:
        ds = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"    [WARN] Failed to load {subject}: {e}")
        return {"accuracy": 0.0, "n_samples": 0, "error": str(e)}

    choices = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices]

    correct = 0
    total = 0

    for item in ds:
        question = item["question"]
        options = item["choices"]
        answer_idx = item["answer"]

        prompt = (
            f"The following is a multiple choice question about {subject.replace('_', ' ')}.\n\n"
            f"{question}\n"
        )
        for i, opt in enumerate(options):
            prompt += f"{choices[i]}. {opt}\n"
        prompt += "Answer:"

        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0, -1, :]
                choice_logits = logits[choice_ids]
                pred = choice_logits.argmax().item()

            if pred == answer_idx:
                correct += 1
            total += 1
        except Exception:
            total += 1

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "n_samples": total,
    }


def evaluate_perplexity(model, tokenizer, dataset: str = "wikitext",
                         config: str = "wikitext-2-raw-v1",
                         max_samples: int = 200) -> float:
    """计算 wikitext perplexity。"""
    try:
        ds = load_dataset(dataset, config, split="test")
        text = "\n\n".join([s for s in ds["text"] if len(s.strip()) > 0][:max_samples])
        encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            outputs = model(**encodings, labels=encodings["input_ids"])
        ppl = float(torch.exp(outputs.loss).item())
        return ppl
    except Exception as e:
        print(f"    [WARN] Perplexity eval failed: {e}")
        return -1.0


# ============================================================
# Experiment
# ============================================================

# 评估的代表性科目（覆盖 STEM/Humanities/Social）
EVAL_SUBJECTS = [
    "abstract_algebra",      # STEM: abstract math
    "machine_learning",       # STEM: applied
    "moral_disputes",        # Humanities: ethics
    "professional_law",       # Humanities: formal
    "us_foreign_policy",     # Social
]

STEM_SET = {
    "abstract_algebra", "anatomy", "astronomy", "college_biology",
    "college_chemistry", "college_computer_science", "college_mathematics",
    "college_physics", "computer_security", "conceptual_physics",
    "electrical_engineering", "elementary_mathematics", "formal_logic",
    "high_school_biology", "high_school_chemistry",
    "high_school_computer_science", "high_school_mathematics",
    "high_school_physics", "high_school_statistics", "machine_learning",
    "medical_genetics", "virology",
}

HUMANITIES_SET = {
    "formal_logic", "high_school_european_history",
    "high_school_us_history", "high_school_world_history",
    "international_law", "jurisprudence", "logical_fallacies",
    "moral_disputes", "moral_scenarios", "philosophy", "prehistory",
    "professional_law", "world_religions",
}

SOCIAL_SET = {
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_microeconomics", "econometrics",
    "human_sexuality", "marketing", "public_relations",
    "security_studies", "sociology", "us_foreign_policy",
}


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--subjects", type=str, default="all",
                        help="Comma-separated or 'all' or 'stem5'")
    args = parser.parse_args()

    # Resolve model path — prefer local 1.5B
    # Check both weight_graph/models and ../models (sibling to weight_graph)
    local_0b5 = Path("models/Qwen2.5-0.5B")
    local_1b5 = Path("models/Qwen2.5-1.5B")
    parent_0b5 = Path("../models/Qwen2.5-0.5B")
    parent_1b5 = Path("../models/Qwen2.5-1.5B")

    for mp, name in [(local_1b5, "Qwen2.5-1.5B (local)"),
                     (parent_1b5, "Qwen2.5-1.5B (parent)"),
                     (local_0b5, "Qwen2.5-0.5B (local)"),
                     (parent_0b5, "Qwen2.5-0.5B (parent)")]:
        if mp.exists():
            model_path = str(mp)
            model_name_display = name
            break
    else:
        model_path = args.model_path
        model_name_display = args.model_path
        model_name_display = args.model_path

    output_dir = Path("results/weight_graph/exp09")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine subjects to evaluate
    if args.subjects == "all":
        subjects_to_eval = EVAL_SUBJECTS
    elif "," in args.subjects:
        subjects_to_eval = [s.strip() for s in args.subjects.split(",")]
    else:
        subjects_to_eval = [args.subjects]

    print(f"[exp09] Model: {model_name_display}")
    print(f"[exp09] Conditions: baseline, hub_490, high_pr_control(#159), random_control(#100)")
    print(f"[exp09] Subjects: {subjects_to_eval}")

    # Define ablation conditions
    # name → list of neuron indices to zero
    conditions = {
        "baseline": [],          # no ablation
        "hub_490": [490],        # Neuron 490 (PageRank #1)
        "high_pr_control": [159], # L23_mlp_in_159 (PageRank rank 4 in L23, same layer as #490)
        "random_control": [100], # Random neuron #100
    }

    results = {}

    # Load existing checkpoint for resume
    checkpoint_path = output_dir / "neuron_ablation_checkpoint.json"
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            saved = json.load(f)
        # Load all saved conditions
        for cond_name_saved, cond_data in saved.get("conditions", {}).items():
            results[cond_name_saved] = cond_data
        print(f"[exp09] Resuming from checkpoint: {list(results.keys())}")

    for cond_name, neuron_indices in conditions.items():
        # Skip if this condition is fully done
        if cond_name in results:
            acc = results[cond_name].get("overall_mean_acc", "N/A")
            ppl = results[cond_name].get("perplexity", "N/A")
            print(f"\n[exp09] === Condition: {cond_name} === [ALREADY DONE] PPL={ppl}, acc={acc}")
            continue

        print(f"\n[exp09] === Condition: {cond_name} ===")
        print(f"\n[exp09] === Condition: {cond_name} ===")
        t0 = time.time()

        # Load fresh model for each condition
        print(f"    Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            padding_side="left",
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
        model.eval()
        print(f"    Model loaded in {time.time()-t0:.1f}s")

        # Apply ablation
        if neuron_indices:
            print(f"    Ablating neurons: {neuron_indices}")
            for ni in neuron_indices:
                n_zeros = zero_neuron_all_layers(model, ni)
                print(f"    Zeroed {n_zeros} param slices for neuron {ni}")
        else:
            print(f"    No ablation (baseline)")

        # Evaluate perplexity
        print(f"    Evaluating perplexity...", end=" ", flush=True)
        ppl = evaluate_perplexity(model, tokenizer)
        print(f"PPL={ppl:.4f}" if ppl > 0 else "PPL=ERROR")

        # Evaluate MMLU subjects — with per-subject checkpointing for crash recovery
        # Restore subject_results from checkpoint if resuming a partial condition
        subject_results = {}
        if cond_name in results:
            saved_per_subject = results[cond_name].get("per_subject", {})
            subject_results = dict(saved_per_subject)
            already_done = set(subject_results.keys())
            print(f"    Restored {len(subject_results)} subject results from checkpoint: {list(already_done)}")

        for subject in subjects_to_eval:
            # Resume: skip if we already have this result from a prior run
            if subject in subject_results:
                print(f"    [skip {subject}] already done")
                continue
            print(f"    Evaluating {subject}...", end=" ", flush=True)
            t_sub = time.time()
            result = evaluate_subject(model, tokenizer, subject)
            elapsed = time.time() - t_sub
            subject_results[subject] = result
            print(f"acc={result['accuracy']:.3f} ({result['n_samples']} samples, {elapsed:.0f}s)")

            # Per-subject streaming checkpoint — crash recovery point
            _write_checkpoint(cond_name, neuron_indices, ppl, subject_results)

        # Categorize results
        def cat_mean(cat_set):
            accs = [subject_results[s]["accuracy"]
                    for s in cat_set if s in subject_results]
            return float(np.mean(accs)) if accs else 0.0

        results[cond_name] = {
            "neurons_ablated": neuron_indices,
            "perplexity": ppl,
            "per_subject": subject_results,
            "category_means": {
                "STEM": cat_mean(STEM_SET),
                "Humanities": cat_mean(HUMANITIES_SET),
                "Social": cat_mean(SOCIAL_SET),
            },
            "overall_mean_acc": float(np.mean([subject_results[s]["accuracy"] for s in subject_results])),
        }

        del model
        print(f"    Condition done in {time.time()-t0:.1f}s total")
        sys.stdout.flush()

        # Streaming checkpoint after each condition
        checkpoint = {
            "model": model_name_display,
            "conditions": {k: v for k, v in results.items()},
            "eval_subjects": subjects_to_eval,
            "conditions_completed": list(results.keys()),
        }
        with open(output_dir / "neuron_ablation_checkpoint.json", "w") as f:
            json.dump(checkpoint, f, indent=2)
            f.flush()
            os.fsync(f.fileno())

    # ============================================================
    # Analysis: compare conditions
    # ============================================================
    print(f"\n[exp09] === Analysis ===")

    baseline = results["baseline"]
    print(f"\nBaseline: PPL={baseline['perplexity']:.4f}, "
          f"mean_acc={baseline['overall_mean_acc']:.4f}")

    for cond_name in ["hub_490", "high_pr_control", "random_control"]:
        r = results[cond_name]
        ablated = r["neurons_ablated"]
        ppl_delta = r["perplexity"] - baseline["perplexity"]
        acc_delta = r["overall_mean_acc"] - baseline["overall_mean_acc"]

        ppl_pct = (ppl_delta / baseline["perplexity"]) * 100 if baseline["perplexity"] > 0 else 0
        print(f"\n{cond_name} (neurons {ablated}):")
        print(f"  PPL: {r['perplexity']:.4f} (delta={ppl_delta:+.4f}, {ppl_pct:+.2f}%)")
        print(f"  Mean acc: {r['overall_mean_acc']:.4f} (delta={acc_delta:+.4f})")
        for cat, cat_acc in r["category_means"].items():
            cat_delta = cat_acc - baseline["category_means"].get(cat, 0)
            print(f"    {cat}: {cat_acc:.4f} (delta={cat_delta:+.4f})")

    # Stop/Go decision
    print(f"\n[exp09] === Stop/Go ===")

    hub = results["hub_490"]
    hp_pr = results["high_pr_control"]
    rnd = results["random_control"]

    hub_ppl_delta = hub["perplexity"] - baseline["perplexity"]
    hp_pr_ppl_delta = hp_pr["perplexity"] - baseline["perplexity"]
    rnd_ppl_delta = rnd["perplexity"] - baseline["perplexity"]

    hub_acc_delta = hub["overall_mean_acc"] - baseline["overall_mean_acc"]
    hp_pr_acc_delta = hp_pr["overall_mean_acc"] - baseline["overall_mean_acc"]
    rnd_acc_delta = rnd["overall_mean_acc"] - baseline["overall_mean_acc"]

    print(f"Hub #490:  PPL_delta={hub_ppl_delta:+.4f}, acc_delta={hub_acc_delta:+.4f}")
    print(f"High-PR #159: PPL_delta={hp_pr_ppl_delta:+.4f}, acc_delta={hp_pr_acc_delta:+.4f}")
    print(f"Random #100: PPL_delta={rnd_ppl_delta:+.4f}, acc_delta={rnd_acc_delta:+.4f}")

    # Decision logic
    if hub_ppl_delta > hp_pr_ppl_delta and hub_acc_delta < hp_pr_acc_delta:
        verdict = "Go: #490 特异性影响大于同层 high-PR 对照"
        go_decision = "GO"
    elif hub_ppl_delta > rnd_ppl_delta * 1.5 and hub_acc_delta < rnd_acc_delta * 0.8:
        verdict = "Hold: #490 影响大于随机对照，但不比特异于同层 high-PR"
        go_decision = "HOLD"
    else:
        verdict = "No-Go: #490 与对照无显著特异性差异"
        go_decision = "NO_GO"

    print(f"\nDecision: {verdict}")
    print(f"Verdict code: {go_decision}")

    # ============================================================
    # Save results
    # ============================================================
    output = {
        "model": model_name_display,
        "conditions": {k: {kk: str(vv) if isinstance(vv, Path) else vv
                           for kk, vv in v.items()}
                       for k, v in results.items()},
        "analysis": {
            "baseline_ppl": float(baseline["perplexity"]),
            "hub_490_ppl_delta": float(hub_ppl_delta),
            "high_pr_ppl_delta": float(hp_pr_ppl_delta),
            "random_ppl_delta": float(rnd_ppl_delta),
            "hub_490_acc_delta": float(hub_acc_delta),
            "high_pr_acc_delta": float(hp_pr_acc_delta),
            "random_acc_delta": float(rnd_acc_delta),
            "go_decision": go_decision,
            "verdict": verdict,
        },
        "eval_subjects": subjects_to_eval,
    }

    output_path = output_dir / "neuron_ablation_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n[exp09] Results saved to {output_path}")


if __name__ == "__main__":
    run()
