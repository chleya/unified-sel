# Control Contract for #490 Activation Profile

**Date**: 2026-04-12
**Status**: Active
**Parent**: FRAGILITY_PRIOR_REALIGNMENT_2026-04-12.md

---

## 1. Control Selection

| Control | Neuron | Justification |
|---------|--------|---------------|
| **Primary (hub)** | **#490** | PageRank 全模型第一，L15-L18+L23 跨层 hub |
| **Same-layer high PageRank** | **#159** | L23_mlp_in，PageRank rank 4 in L23，同层对照 |
| **Random baseline** | **#100** | 随机选择，与 PageRank 无关 |

---

## 2. Metrics

### Primary Metrics

| Metric | Definition | Measurement |
|--------|------------|-------------|
| `activation_mean` | 平均激活值（post-down_proj，last token） | forward hook |
| `activation_std` | 激活值标准差 | forward hook |
| `activation_abs_mean` | 激活值绝对值均值 | forward hook（稀疏性指标） |
| `activation_max` | 最大激活值 | forward hook |

### Derived Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| `sparsity_ratio` | fraction of activations near zero (|a| < 1e-4) | 是否为稀疏激活 |
| `domain_sensitivity` | variance of mean activation across domains | 是否有域特异性 |
| `layer_peak` | layer with maximum activation | 激活集中在哪层 |

---

## 3. Pass/Fail Logic (Fragility Gate)

### Gate W-A: Neuron Distinctiveness

**Condition**: `#490` activation profile must be statistically distinguishable from BOTH controls.

- Method: `#490` mean activation vs control mean activation, per-domain
- Threshold: `#490` mean differs from control by > 2σ of the control distribution, OR `#490` mean differs from random by > 3σ
- Pass: **DA** (Distinctive Activation)
- Fail: **NA** (Not Distinctive)

### Gate W-B: Domain Sensitivity

**Condition**: `#490` activation pattern differs meaningfully across input domains.

- Threshold: activation variance across STEM/Humanities/Social > 0.05
- Pass: **DS** (Domain Sensitive)
- Fail: **NS** (Not Sensitive)

### Gate W-C: Stability

**Condition**: Profile is reproducible across prompt variations.

- Threshold: std of mean activation across prompt variations < 0.5
- Pass: **STABLE**
- Fail: **VARIABLE**

---

## 4. Decision Table

| DA | DS | STABLE | Verdict |
|----|----|--------|---------|
| Yes | Yes | Yes | **Go** — Profile is distinctive, domain-sensitive, and stable |
| Yes | Yes | No | **Hold** — Distinctive and sensitive but unstable |
| Yes | No | Yes | **Hold** — Distinctive but not domain-specific |
| No | Any | Any | **Stop** — Not distinctive from controls |

---

## 5. Output Schema

```json
{
  "neuron": 490,
  "global_stats": {
    "activation_mean": 0.0,
    "activation_std": 0.0,
    "activation_abs_mean": 0.0,
    "activation_max": 0.0,
    "per_layer": {"layer_name": value}
  },
  "domain_STEM": { ... },
  "domain_Humanities": { ... },
  "domain_Social": { ... },
  "domain_Math": { ... },
  "gates": {
    "DA": "PASS|FAIL",
    "DS": "PASS|FAIL",
    "STABLE": "PASS|FAIL"
  },
  "verdict": "GO|HOLD|STOP"
}
```

---

## 6. Experiment Parameters

- Model: Qwen2.5-1.5B (local)
- Input sets: synthetic domain prompts + MMLU questions
- Layers: all MLP layers (L0-L23)
- Hook point: MLP down_proj output (post-activation, pre-residual-add)
- Prompt pool: ~30 prompts across 4 domains

---

## 7. Commands

```powershell
# W4: Run activation profile
python F:\unified-sel\weight_graph\experiments\exp10_activation_profile.py --model_path Qwen/Qwen2.5-0.5B
```

Results: `results/weight_graph/exp10/activation_profile.json`
