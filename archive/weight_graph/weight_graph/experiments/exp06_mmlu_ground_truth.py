"""
实验 6：MMLU Per-Subject Accuracy Ground Truth

目标：获取 Qwen 模型在 MMLU 57 个 subject 上的逐科准确率。
这是 Phase 2c（逐层拓扑 × 任务性能关联）的前置条件。

方法：4-choice multiple choice evaluation
预期耗时：约 3-4 小时（CPU 推理，57 subjects × ~100 题）
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# MMLU 57 subjects
MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics",
    "clinical_knowledge", "college_biology", "college_chemistry",
    "college_computer_science", "college_mathematics", "college_medicine",
    "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics",
    "formal_logic", "global_facts", "high_school_biology",
    "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography",
    "high_school_government_and_politics", "high_school_macroeconomics",
    "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology",
    "high_school_statistics", "high_school_us_history",
    "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies",
    "machine_learning", "management", "marketing", "medical_genetics",
    "miscellaneous", "moral_disputes", "moral_scenarios", "nutrition",
    "philosophy", "prehistory", "professional_accounting",
    "professional_law", "professional_medicine", "professional_psychology",
    "public_relations", "security_studies", "sociology",
    "us_foreign_policy", "virology", "world_religions",
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


def evaluate_subject(model, tokenizer, subject: str) -> Dict:
    """评估单个 MMLU subject，返回 accuracy 和样本数。"""
    try:
        ds = load_dataset("cais/mmlu", subject, split="test")
    except Exception as e:
        print(f"  [WARN] Failed to load {subject}: {e}")
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
            total += 1  # count even failed attempts

    return {
        "accuracy": correct / total if total > 0 else 0.0,
        "n_samples": total,
    }


def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--subjects", type=str, default="all",
                        help="Comma-separated list or 'all' or 'stem5'")
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    # Resolve model path - prefer local 1.5B if available
    local_0b5 = Path("models/Qwen2.5-0.5B")
    local_1b5 = Path("models/Qwen2.5-1.5B")

    # Try local paths first
    if local_1b5.exists():
        model_path = str(local_1b5)
        model_name_display = "Qwen2.5-1.5B (local)"
    elif local_0b5.exists():
        model_path = str(local_0b5)
        model_name_display = "Qwen2.5-0.5B (local)"
    else:
        model_path = args.model_path
        model_name_display = args.model_path

    print(f"[exp06] Model: {model_name_display}")
    print(f"[exp06] Loading model and tokenizer...")

    t0 = time.time()
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
    print(f"[exp06] Model loaded in {time.time()-t0:.1f}s")

    # Determine subjects to run
    if args.subjects == "all":
        subjects_to_run = MMLU_SUBJECTS
    elif args.subjects == "stem5":
        subjects_to_run = list(STEM_SET)[:5]
    elif "," in args.subjects:
        subjects_to_run = [s.strip() for s in args.subjects.split(",")]
    else:
        subjects_to_run = [args.subjects]

    print(f"[exp06] Running {len(subjects_to_run)} subjects: {subjects_to_run[:5]}...")

    results = {}
    category_results = {"STEM": [], "Humanities": [], "Social": [], "Other": []}

    output_dir = Path("results/weight_graph/exp06")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "mmlu_checkpoint.json"
    progress_path = output_dir / "progress.txt"

    for i, subject in enumerate(subjects_to_run):
        print(f"  [{i+1}/{len(subjects_to_run)}] {subject}...", end=" ", flush=True)
        t0 = time.time()
        result = evaluate_subject(model, tokenizer, subject)
        elapsed = time.time() - t0
        results[subject] = result
        print(f"acc={result['accuracy']:.3f} ({result['n_samples']} samples, {elapsed:.0f}s)")
        sys.stdout.flush()

        # Categorize
        if subject in STEM_SET:
            category_results["STEM"].append(result["accuracy"])
        elif subject in HUMANITIES_SET:
            category_results["Humanities"].append(result["accuracy"])
        elif subject in SOCIAL_SET:
            category_results["Social"].append(result["accuracy"])
        else:
            category_results["Other"].append(result["accuracy"])

        # Stream checkpoint: write after each subject
        per_subject_data = {
            s: {"accuracy": results[s]["accuracy"], "n_samples": results[s]["n_samples"]}
            for s in results
        }
        all_accuracies = [results[s]["accuracy"] for s in results]
        overall = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0
        by_category = {}
        for cat, accs in category_results.items():
            by_category[cat] = {"avg_accuracy": sum(accs) / len(accs) if accs else 0.0, "n_subjects": len(accs)}
        checkpoint = {
            "model": model_name_display,
            "per_subject": per_subject_data,
            "by_category": by_category,
            "overall_accuracy": overall,
            "n_subjects_evaluated": i + 1,
        }
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        with open(progress_path, "w") as f:
            f.write(f"{i + 1}\n{len(subjects_to_run)}\n{subject}\n")
            f.flush()
            os.fsync(f.fileno())
        sys.stdout.flush()


    # Compute summaries
    per_subject_data = {
        s: {
            "accuracy": results[s]["accuracy"],
            "n_samples": results[s]["n_samples"],
        }
        for s in subjects_to_run
    }

    all_accuracies = [results[s]["accuracy"] for s in subjects_to_run]
    overall = sum(all_accuracies) / len(all_accuracies) if all_accuracies else 0.0

    by_category = {}
    for cat, accs in category_results.items():
        by_category[cat] = {
            "avg_accuracy": sum(accs) / len(accs) if accs else 0.0,
            "n_subjects": len(accs),
        }

    output = {
        "model": model_name_display,
        "per_subject": per_subject_data,
        "by_category": by_category,
        "overall_accuracy": overall,
        "n_subjects_evaluated": len(subjects_to_run),
    }

    # Rename checkpoint → final output
    output_path = output_dir / "mmlu_per_subject.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # Mark progress done
    with open(progress_path, "w") as f:
        f.write(f"{len(subjects_to_run)}\n{len(subjects_to_run)}\ndone\n")
        f.flush()
        os.fsync(f.fileno())

    print(f"\n[exp06] === Results ===")
    print(f"  Overall accuracy: {overall:.4f}")
    for cat, data in by_category.items():
        print(f"  {cat}: {data['avg_accuracy']:.4f} ({data['n_subjects']} subjects)")
    print(f"[exp06] Saved to {output_path}")
    sys.stdout.flush()

    # Save summary to REPORT.md
    summary_path = output_dir / "mmlu_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Model: {model_name_display}\n")
        f.write(f"Overall: {overall:.4f}\n")
        for cat, data in by_category.items():
            f.write(f"{cat}: {data['avg_accuracy']:.4f}\n")
    print(f"[exp06] Summary saved to {summary_path}")


if __name__ == "__main__":
    run()
