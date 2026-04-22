# Unified-SEL 项目总览与整理索引

更新时间：2026-04-22

这份文档用于快速理解 `F:\unified-sel` 当前到底是什么、哪些结论可靠、哪些目录还在使用、哪些只是历史实验或材料沉淀。

---

## 一句话介绍

`unified-sel` 最初是一个“无显式任务边界的 continual learning / endogenous structure formation”研究项目，后来经过实验审计后转向两条更可靠的主线：

1. **Capability Router 工具线**：判断小模型输出应该直接接受、验证、修订还是升级。
2. **Boundary-local amplification 论文线**：证明 feedback retry 的收益集中在边界附近，而不是普遍增强。

旧核心假设：

> surprise-driven structural birth/death 能显著优于 EWC

当前状态：**已降级为未验证机制假设**，不能作为主结论宣传。

---

## 当前项目结构判断

| 方向 | 当前状态 | 结论 |
|---|---|---|
| Capability Router | 主产品线 | 继续推进 |
| Boundary-local amplification | 主论文线 | 可写 short paper |
| TopoMem OBD | 次级健康监控线 | 可继续验证 batch-level drift |
| TopoMem per-task routing | 已拒绝 | 不再作为 routing monitor |
| surprise > EWC | 未验证 | 不作为主张 |
| self-aware-llm | 未来叙事 | 现在不作为构建目标 |
| meta-controller | 旁支探索 | 可归档为后续研究线 |
| CEP-CC | 已拆分 | 独立项目：`F:\cep-cc` |
| weight_graph | 旁支分析 | 作为权重拓扑/任务签名参考 |

---

## 可靠结论与禁止过度声明

### 可以安全声明

- Boundary-local amplification 存在：Phase A，`p=0.0008`。
- ABOVE-zone filtering 能减少 feedback 调用：Phase E，约 `54.4%`。
- Capability Router 的 semantic / counterfactual monitors 在当前 benchmark 上有效。
- Qwen2.5-0.5B 的自报 confidence 不可靠：confidence 高但实际成功率低。
- 0.5B LLM revision 在 code-20 上无效：`0/17` 修复成功。
- TopoMem 可以作为 batch-level deployment health / drift monitor 候选。

### 不能声明

- 不能说 Unified-SEL 已经击败 EWC。
- 不能说 TopoMem surprise 能判断单题答案是否正确。
- 不能把 `cost_units` 当真实成本。
- 不能把 OracleSolver 的 100% escalation success 当真实大模型结果。
- 不能把 self-aware LLM 说成已验证系统。
- 不能把 public benchmark 文件泄漏 hidden tests / fixed code / expected route。

---

## 目录整理

### `core/`

核心代码目录。

| 文件 | 作用 |
|---|---|
| `capability_benchmark.py` | Capability Router 主引擎：任务、solver、verifier、monitor、policy、BatchHealthMonitor |
| `llm_solver.py` | Qwen / llama.cpp solver 适配层 |
| `structure.py` | 旧 Unified-SEL Structure：surprise / tension / utility |
| `pool.py` | 旧结构池生命周期：reinforce / branch / create |
| `learner.py` | DFA / SEL-Lab 风格 learner |
| `runtime.py` | 路径、保存、缓存工具 |
| `topo_fusion.py` | TopoMem / topology fusion 相关实验代码 |
| `experiment_config.py` | 实验配置 |
| `experiment_utils.py` | 实验工具 |

当前最重要文件：`core/capability_benchmark.py` 和 `core/llm_solver.py`。

### `experiments/capability/`

Capability Router 当前主线。

| 文件 | 作用 |
|---|---|
| `capbench.py` | CLI：run / compare / report / list-monitors / list-policies |
| `export_bench.py` | 导出 public/eval JSONL benchmark |
| `README.md` | Capability Router 使用说明 |
| `solver_compare.py` | synthetic solver 与 real LLM solver 对比 |
| `llm_routing_experiment.py` | real LLM routing policy 对比 |
| `prompt_ab_test.py` | prompt engineering A/B |
| `llm_revision_test.py` | LLM revision 能力测试 |
| `topomem_obd_preflight.py` | TopoMem OBD 初步验证 |
| `topomem_obd_multiseed.py` | TopoMem OBD 多种子验证 |

最新策略：

- `monitor_repair_triage`：适合可修订 solver。
- `monitor_no_revision_triage`：适合弱修订 solver，例如 Qwen2.5-0.5B。

### `data/capability_boundary_bench/`

Capability benchmark 数据。

| 文件 | 说明 |
|---|---|
| `code-20.public.jsonl` | 对外 public code benchmark，不含 hidden tests |
| `code-20.eval.jsonl` | 内部 eval code benchmark，含 hidden tests |
| `mixed-40.public.jsonl` | 对外 public mixed benchmark |
| `mixed-40.eval.jsonl` | 内部 eval mixed benchmark |
| `README.md` | 泄漏边界说明 |

不要把 `.eval.jsonl` 当 public 数据发布。

### `double_helix/`

Boundary-local amplification 论文线实验目录。

主要用途：

- feedback retry / blind retry / boundary zone 实验
- ABOVE / NEAR / BELOW 区分
- artifact audit
- cross-solver / group-kfold / no-leak validation

关键结论：

- feedback 的收益集中在 near-boundary。
- ABOVE 任务可过滤。
- NEAR/BELOW discrimination 仍弱，不能作为部署策略。

### `topomem/`

TopoMem 子系统。

当前定位：

- **不是 per-task answer routing monitor**
- 是 batch-level deployment health / OBD 候选

重要模块：

| 文件 | 作用 |
|---|---|
| `memory.py` | 记忆存储与检索 |
| `topology.py` | 拓扑 / H1-H2 / 图结构指标 |
| `health_controller.py` | 健康监控控制器 |
| `self_awareness.py` | self-aware 叙事原型，当前不作为已验证系统 |
| `adapters.py` | adapter / surprise / tension 相关机制 |

### `weight_graph/`

LLM 权重拓扑分析旁支。

用途：

- 权重矩阵图构建
- PageRank / topology feature
- 可能作为 future task signature 或 zero-cost routing feature 参考

不是当前主线。

### `experiments/meta_controller/` 与 `META_CONTROLLER_*`

Meta-controller 旁支探索。

定位：

- 可作为“胶水项目”或后续控制器设计材料。
- 当前不是主线结论来源。
- 不能混入 Capability Router 主结论，除非重新验证。

### `experiments/cep_cc/` 与 `CEP_CC_*`

CEP-CC / communication-compression 旁支。

定位：

- 已拆分为独立项目：`F:\cep-cc`。
- 可作为未来组合系统或胶水项目材料。
- 当前不影响 Capability Router 和 boundary-local amplification 的主结论。
- `unified-sel/archive/cep_cc/README.md` 记录拆分指针。

### `analysis/`

旧分析脚本目录。

用途：

- variance diagnostics
- boundary diagnostics
- 旧 continual learning 结果分析

属于历史与辅助分析。

### `results/`

所有实验输出。

规则：

- 不手工编辑 JSON 结果。
- 新实验应自动写入此目录。
- 报告中引用结果时必须说明数据来源、seed、cost model、oracle assumption。

### `tests/`

测试目录。

当前关键入口：

```bash
python tests/smoke_test.py
```

每次代码变更后都应运行。

---

## 顶层文档索引

优先阅读顺序：

1. `AGENTS.md`：当前规则、红线、工作协议。
2. 本文件：项目总览和目录索引。
3. `PROJECT_PIVOT_DECISION_2026-04-16.md`：为什么项目转向。
4. `experiments/capability/README.md`：Capability Router 使用说明。
5. `TOPOMEM_ROUTING_MONITOR_RESULT_2026-04-16.md`：为什么 TopoMem 不进 per-task routing core。
6. `CAPABILITY_BENCHMARK_TOOLKIT_PLAN.md`：Capability toolkit 路线。
7. `EXPERIMENT_LOG.md`：完整实验记录。

注意：`STATUS.md` 当前含有大量编码损坏文本，但仍包含一些最新状态；优先以 `AGENTS.md`、本文件和具体结果文件交叉确认。

---

## 常用命令

### Smoke test

```bash
cd F:\unified-sel
python tests/smoke_test.py
```

### Capability Router

列出 monitors：

```bash
python experiments/capability/capbench.py list-monitors
```

列出 policies：

```bash
python experiments/capability/capbench.py list-policies
```

运行 synthetic benchmark：

```bash
python experiments/capability/capbench.py run --suite mixed --protocol monitor_repair_triage --routing-monitor semantic --num-tasks 40 --seed 7
```

弱修订 LLM 推荐策略 sanity run：

```bash
python experiments/capability/capbench.py run --suite code --protocol monitor_no_revision_triage --routing-monitor semantic --num-tasks 20 --seed 7
```

生成 report：

```bash
python experiments/capability/capbench.py report results/capability_benchmark/<result>.json
```

### Real LLM 路由实验

前提：本地 llama.cpp server 运行在 `http://127.0.0.1:8081`。

```bash
python experiments/capability/llm_routing_experiment.py
```

当前状态：最近检查时 server 不可达，因此未跑新的 real-LLM routing comparison。

---

## 当前主线下一步

### 1. Capability Router

最高优先级：

- 启动 llama.cpp server。
- 运行 `llm_routing_experiment.py`。
- 比较：
  - `monitor_gate`
  - `monitor_repair_triage`
  - `monitor_no_revision_triage`
  - `verifier_first`

目标：确认 Qwen2.5-0.5B 场景下 no-revision triage 是否比 revision policy 更经济。

### 2. TopoMem OBD

下一步应该是 batch-level health validation，而不是 per-task routing。

建议：

- code -> code control
- code -> reasoning domain shift
- trivial -> hard gradual shift
- 比较 centroid drift、pairwise similarity、H1/H2、predictive residual

### 3. Boundary-local amplification paper

短论文可写，但必须保持诚实边界：

- 强调 inverted-U / ABOVE filtering。
- 明确 NEAR/BELOW deployable discrimination 未解决。
- 明确 patch_size / bug_type artifact audit。
- 不使用 assumed cost model 做真实成本结论。

### 4. 胶水项目方向

如果继续整合 SEL-Lab / SDAS / fcrs_mis / TopoMem / Capability Router，建议统一到一个薄层：

```text
Predictive State / Health Adapter
```

它输出：

```json
{
  "latent_residual": "...",
  "distribution_drift": "...",
  "normality_deviation": "...",
  "topology_shift": "...",
  "health_status": "...",
  "routing_prior": "..."
}
```

用途：

- SEL-Lab 提供 task signature。
- SDAS 使用 predictive residual 触发结构动作。
- fcrs_mis 管理结构池与 utility。
- TopoMem 提供 geometry health。
- Capability Router 使用 health prior 调整 verify / escalate 策略。
- CEP-CC 已独立为连续通信协议涌现研究项目，可作为未来 communication module 接入，不再放入 `unified-sel` 主线。

---

## 建议清理规则

暂不建议删除文件。当前仓库是研究型工作区，很多文档和结果有溯源价值。

建议先做逻辑分层：

| 类别 | 处理 |
|---|---|
| 当前主线代码 | 保留并继续测试 |
| 当前主线文档 | 保留，必要时修编码 |
| 历史机制实验 | 标注 archived，不删除 |
| 旁支探索 | 标注 branch / sidecar |
| CEP-CC | 已拆分到 `F:\cep-cc`，`unified-sel` 先保留原文件到下一轮显式清理 |
| 大模型文件与 cache | 不移动，除非确认路径依赖 |
| `results/` | 不手工编辑 |
| `STATUS.md` | 后续可重写为 UTF-8，但需先备份 |

---

## 快速接手结论

如果只做一件事：

> 继续 Capability Router real-LLM 验证，优先比较 `monitor_no_revision_triage` 是否适合 Qwen2.5-0.5B。

如果做第二件事：

> 把 TopoMem OBD 作为 batch-level deployment health monitor 继续验证，不要再把 TopoMem surprise 当 per-task routing monitor。

如果写论文：

> 写 boundary-local amplification short paper，主打 inverted-U、ABOVE filtering 和 artifact audit。
