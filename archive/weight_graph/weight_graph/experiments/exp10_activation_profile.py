"""
实验 10：Hub Neuron 490 Activation Profile 分析

目标：在 #490 ablation 的因果证据基础上，理解其激活模式特征。

设计：
  - 对 #490, #159(high_pr), #100(random) 三个 neuron
  - 在 baseline 模型上，按 layer 提取各 neuron 的激活强度分布
  - 按 input domain（STEM/Humanities/Social/Math）分组，看激活是否有域差异

激活提取方法：
  - forward hook 提取 MLP layer 输出（post-down_proj）
  - 记录 neuron index 490/159/100 在各层的激活值
  - 统计量：mean, std, sparsity, max_activation

预计耗时：~30 分钟（单次 forward，无 ablation，固定输入集）
"""

from __future__ import annotations

import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ============================================================
# Activation extraction via hooks
# ============================================================

class NeuronActivationCollector:
    """用 forward hook 收集指定 neuron index 的激活值。"""

    def __init__(self, model, tokenizer, neuron_indices: List[int]):
        self.model = model
        self.tokenizer = tokenizer
        self.neuron_indices = neuron_indices
        self.hooks = []
        self.activations = {ni: [] for ni in neuron_indices}
        self.layer_names = []

    def register_hooks(self):
        """注册所有 MLP 层的 down_proj 输出 hook。"""
        def make_hook(neuron_idx, layer_name):
            def hook(module, input, output):
                # output shape: [batch, seq_len, hidden_dim]
                acts = output[0, -1, neuron_idx].detach().cpu().numpy()
                self.activations[neuron_idx].append(acts)
                if layer_name not in self.layer_names:
                    self.layer_names.append(layer_name)
            return hook

        for name, module in self.model.named_modules():
            if "mlp.down_proj" in name or "mlp.post_proj" in name:
                for ni in self.neuron_indices:
                    h = module.register_forward_hook(make_hook(ni, name))
                    self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def run(self, texts: List[str]) -> Dict:
        """对每个 text 做一次 forward，返回激活统计。"""
        self.activations = {ni: [] for ni in self.neuron_indices}
        self.layer_names = []

        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
            with torch.no_grad():
                self.model(**inputs)
            # activations accumulate via hooks

        return self.get_stats()

    def get_stats(self) -> Dict:
        """计算每层、每 neuron 的统计量。"""
        stats = {}
        for ni in self.neuron_indices:
            acts_list = self.activations[ni]  # list of scalars per layer
            if not acts_list:
                continue
            acts = np.array(acts_list)
            stats[ni] = {
                "mean": float(np.mean(acts)),
                "std": float(np.std(acts)),
                "max": float(np.max(acts)),
                "min": float(np.min(acts)),
                "abs_mean": float(np.mean(np.abs(acts))),
                "per_layer": {ln: float(acts_list[i]) for i, ln in enumerate(self.layer_names)},
            }
        return stats


# ============================================================
# Test prompts by domain
# ============================================================

DOMAIN_PROMPTS = {
    "STEM": [
        "Solve for x: 2x + 5 = 15. What is x?",
        "What is the derivative of x^2 with respect to x?",
        "If a car accelerates at 2 m/s^2 for 5 seconds, what distance does it travel?",
        "What is the atomic number of carbon?",
        "Explain how photosynthesis works in simple terms.",
    ],
    "Humanities": [
        "What did the Treaty of Versailles demand of Germany after WWI?",
        "In your opinion, what is the most important moral principle?",
        "Describe the political system of ancient Rome.",
        "What is the difference between deontology and consequentialism?",
        "Analyze the themes of power and isolation in 1984.",
    ],
    "Social": [
        "What factors influence consumer demand for a normal good?",
        "Explain the relationship between inflation and unemployment according to the Phillips curve.",
        "What are the main functions of the Federal Reserve System?",
        "How does group polarization affect political opinions?",
        "What is the relationship between GDP and economic well-being?",
    ],
    "Math": [
        "Calculate: 347 + 892 = ?",
        "What is 15% of 240?",
        "If a rectangle has width 8 and length 12, what is its area?",
        "Solve: 3(x - 2) = 12. What is x?",
        "What is the square root of 144?",
    ],
}

MMLU_STEM_QUESTIONS = [
    # abstract_algebra
    "Let G be a group and let H be a subgroup of G. If |G| = 12 and |H| = 4, what is the index of H in G?",
    # machine_learning
    "What is the primary difference between supervised and unsupervised learning?",
    # college_physics
    "A 2 kg object is dropped from a height of 10 m. What is its velocity just before hitting the ground?",
    # college_chemistry
    "How many moles of solute are present in 2 liters of a 0.5 M solution?",
    # computer_security
    "What is the main vulnerability of a buffer overflow attack?",
]

MMLU_HUMANITIES_QUESTIONS = [
    # moral_disputes
    "Is it morally permissible to lie to protect someone from harm?",
    # philosophy
    "What is Descartes' cogito ergo sum and what does it imply about the nature of knowledge?",
    # professional_law
    "Under US law, what is required to establish a valid contract?",
]

MMLU_SOCIAL_QUESTIONS = [
    # econometrics
    "What is the difference between endogeneity and heteroskedasticity in regression analysis?",
    # us_foreign_policy
    "What are the main objectives of US foreign policy regarding nuclear proliferation?",
]


def build_prompt_pool():
    """构建所有测试用 prompts。"""
    prompts = []

    # Domain prompts
    for domain, qs in DOMAIN_PROMPTS.items():
        for q in qs:
            prompts.append({"text": q, "domain": domain, "source": "synthetic"})

    # MMLU prompts
    for q in MMLU_STEM_QUESTIONS:
        prompts.append({"text": q, "domain": "STEM", "source": "mmlu"})
    for q in MMLU_HUMANITIES_QUESTIONS:
        prompts.append({"text": q, "domain": "Humanities", "source": "mmlu"})
    for q in MMLU_SOCIAL_QUESTIONS:
        prompts.append({"text": q, "domain": "Social", "source": "mmlu"})

    return prompts


# ============================================================
# Main experiment
# ============================================================

def run():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    args = parser.parse_args()

    # Resolve model path
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

    output_dir = Path("results/weight_graph/exp10")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Target neurons: 490 (hub), 159 (high_pr control), 100 (random control)
    target_neurons = [490, 159, 100]

    print(f"[exp10] Model: {model_name_display}")
    print(f"[exp10] Target neurons: {target_neurons}")

    # Load model
    print(f"  Loading model...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32, trust_remote_code=True)
    model.eval()
    print(f"  Model loaded in {time.time()-t0:.1f}s")

    # Build prompt pool
    prompts = build_prompt_pool()
    texts = [p["text"] for p in prompts]
    print(f"[exp10] Running {len(texts)} prompts across {len(set(p['domain'] for p in prompts))} domains")

    # Collect activations
    collector = NeuronActivationCollector(model, tokenizer, target_neurons)
    collector.register_hooks()

    print(f"  Collecting activations...")
    t1 = time.time()
    all_stats = collector.run(texts)
    print(f"  Done in {time.time()-t1:.1f}s")
    collector.remove_hooks()

    # Organize by domain
    results = {
        "model": model_name_display,
        "neurons": target_neurons,
        "n_prompts": len(texts),
        "global_stats": all_stats,
    }

    # Per-domain breakdown
    for domain in ["STEM", "Humanities", "Social", "Math"]:
        domain_texts = [prompts[i]["text"] for i in range(len(prompts)) if prompts[i]["domain"] == domain]
        if not domain_texts:
            continue
        collector2 = NeuronActivationCollector(model, tokenizer, target_neurons)
        collector2.register_hooks()
        domain_stats = collector2.run(domain_texts)
        collector2.remove_hooks()
        results[f"domain_{domain}"] = domain_stats

    # Per-layer analysis for key neurons
    print(f"\n[exp10] === Activation Profile Summary ===")
    for ni in target_neurons:
        if ni not in all_stats:
            continue
        s = all_stats[ni]
        label = "hub_490" if ni == 490 else ("high_pr_159" if ni == 159 else "random_100")
        print(f"  Neuron {ni} ({label}): mean={s['mean']:.4f}, std={s['std']:.4f}, |mean|={s['abs_mean']:.4f}")

    # Save
    output_path = output_dir / "activation_profile.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[exp10] Saved to {output_path}")

    # Comparison table
    print(f"\n[exp10] === Control Comparison ===")
    print(f"{'Neuron':12s} {'mean':>10s} {'std':>10s} {'|mean|':>10s} {'max':>10s}")
    for ni in target_neurons:
        if ni not in all_stats:
            continue
        s = all_stats[ni]
        label = "hub_490" if ni == 490 else ("high_pr_159" if ni == 159 else "random_100")
        print(f"  {label:12s} {s['mean']:10.4f} {s['std']:10.4f} {s['abs_mean']:10.4f} {s['max']:10.4f}")

    del model
    print(f"[exp10] Done.")


if __name__ == "__main__":
    run()
