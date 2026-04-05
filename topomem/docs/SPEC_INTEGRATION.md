# SPEC_INTEGRATION.md — 端到端集成与基准测试规格

> **对应模块**：`topomem/system.py` + `topomem/benchmarks/*.py`  
> **对应阶段**：Phase 6（最终阶段）  
> **前置依赖**：所有前序模块  
> **被依赖方**：无（顶层入口）

---

## 1. TopoMemSystem 集成架构

### 1.1 类定义

```python
class TopoMemSystem:
    """TopoMem Reasoner v0.1 完整系统。
    
    这是唯一的用户入口。所有内部模块通过此类协调。
    """
    
    def __init__(self, config: Optional[TopoMemConfig] = None):
        """
        初始化顺序（严格按依赖关系）：
        1. config = config or TopoMemConfig()
        2. self.embedding = EmbeddingManager(config.embedding)
        3. self.topology = TopologyEngine(config.topology)
        4. self.memory = MemoryGraph(config.memory, self.embedding)
        5. self.engine = ReasoningEngine(config.engine)
        6. self.self_aware = SelfAwareness(config.awareness)
        7. self.guard = ConsistencyGuard(config.awareness)
        8. self.adapters = AdapterPool(config.adapter, self.embedding)
        9. self._step = 0
        10. self._process_log: List[ProcessResult] = []
        """
    
    def process(self, input_text: str) -> ProcessResult:
        """完整的输入处理流程。详见 §2。"""
    
    def get_status(self) -> SystemStatus:
        """系统状态快照。详见 §3。"""
    
    def save(self, path: str) -> None:
        """持久化整个系统状态。
        保存内容：memory graph, adapter pool, self_awareness state, config
        """
    
    def load(self, path: str) -> None:
        """从磁盘恢复系统。"""
    
    def reset(self) -> None:
        """清空所有状态，回到初始状态。"""
```

### 1.2 ProcessResult 数据结构

```python
@dataclass
class ProcessResult:
    """一次 process() 调用的完整输出。"""
    
    # 核心输出
    response_text: str                    # LLM 生成的回答
    
    # 检索信息
    retrieved_memories: List[MemoryNode]  # 被检索到的记忆
    retrieval_strategy: str               # 使用的检索策略
    
    # 适配信息
    adapter_used: str                     # 使用的 adapter 名称
    surprise_score: float                 # 输入的意外度
    
    # 记忆写入
    memory_accepted: bool                 # 新知识是否被接受
    memory_reject_reason: Optional[str]   # 拒绝原因（如有）
    
    # 自我认知
    drift_status: Optional[str]           # 当前漂移状态
    calibrated: bool                      # 本次是否触发了校准
    
    # 性能
    latency_ms: float                     # 总耗时（毫秒）
    step: int                             # 第几步

@dataclass
class SystemStatus:
    """系统状态快照。"""
    
    step: int
    memory_node_count: int
    memory_cluster_count: int
    adapter_count: int
    drift_status: str
    last_calibration_step: int
    ram_usage_mb: float
    fingerprint: Optional[np.ndarray]
```

---

## 2. process() 完整流程伪代码

```python
def process(self, input_text: str) -> ProcessResult:
    start_time = time.time()
    self._step += 1
    
    # ---- Step 1: 编码输入 ----
    query_embedding = self.embedding.encode(input_text)
    
    # ---- Step 2: 选择 adapter ----
    adapter, surprise = self.adapters.select_adapter(
        query_embedding, self.memory
    )
    
    # ---- Step 3: 检索记忆 ----
    strategy = "hybrid"  # 默认使用 hybrid
    retrieved = self.memory.retrieve(
        query_embedding, 
        strategy=strategy, 
        k=self.config.memory.similarity_top_k
    )
    
    # ---- Step 4: 推理生成 ----
    system_prompt = adapter.apply(self.engine, input_text)
    response = self.engine.generate(
        prompt=input_text,
        context=retrieved,
        adapter=adapter
    )
    
    # ---- Step 5: 知识提取与守护 ----
    new_knowledge = extract_knowledge(input_text, response, self.engine)
    memory_accepted = False
    reject_reason = None
    
    if new_knowledge:
        knowledge_embedding = self.embedding.encode(new_knowledge)
        accepted, reason = self.guard.should_accept_memory(
            new_knowledge, knowledge_embedding,
            self.memory, self.self_aware, self.topology
        )
        if accepted:
            self.memory.add_memory(
                content=new_knowledge,
                embedding=knowledge_embedding,
                metadata={"source": "process", "step": self._step}
            )
            memory_accepted = True
        else:
            reject_reason = reason
    
    # ---- Step 6: Adapter 反馈 ----
    self.adapters.evolve_adapter(adapter.adapter_id, feedback=0.5)  # MVP: 中性反馈
    
    # ---- Step 7: Surprise 驱动的 adapter 创建 ----
    tension = compute_tension(self.self_aware)
    if surprise > 0.7 and tension < self.config.awareness.drift_threshold:
        # 新领域出现 + 系统稳定 → 尝试创建新 adapter
        cluster_memories = self.memory.retrieve_by_cluster(
            # 找到 query 最近的簇
            self._nearest_cluster(query_embedding)
        )
        if len(cluster_memories) >= 5:
            self.adapters.create_adapter(
                cluster_id=cluster_memories[0].cluster_id,
                representative_memories=cluster_memories[:10],
                engine=self.engine
            )
    
    # ---- Step 8: 自我认知更新 ----
    drift_status = None
    calibrated = False
    if self.self_aware.should_calibrate():
        report = self.self_aware.calibrate(
            self.memory, self.topology, self.engine
        )
        drift_status = report.drift.status
        calibrated = True
    
    # ---- 构造结果 ----
    elapsed_ms = (time.time() - start_time) * 1000
    result = ProcessResult(
        response_text=response,
        retrieved_memories=retrieved,
        retrieval_strategy=strategy,
        adapter_used=adapter.name if hasattr(adapter, 'name') else "default",
        surprise_score=surprise,
        memory_accepted=memory_accepted,
        memory_reject_reason=reject_reason,
        drift_status=drift_status,
        calibrated=calibrated,
        latency_ms=elapsed_ms,
        step=self._step
    )
    self._process_log.append(result)
    return result
```

---

## 3. 四组基准测试详细设计

### Benchmark 1: 知识一致性测试 (`benchmarks/knowledge_consistency.py`)

```
目标：验证系统能在多领域混合输入下保持知识的准确性。

语料准备：
  5 个领域，每领域 20 条知识事实（共 100 条）
  领域建议：
    1. Programming: Python 语法、算法概念
    2. Physics: 基本定律、公式
    3. History: 重要事件、年代
    4. Cooking: 食谱步骤、食材搭配
    5. Geography: 国家首都、地理特征

  另准备 50 个测试问题（每领域 10 个），问题的答案在输入的知识中。

实验流程：
  1. 系统重置
  2. 逐条 process() 输入 100 条知识（随机打乱顺序）
  3. 逐条 process() 提出 50 个问题
  4. 记录每个问题的回答

评估指标：
  - accuracy: 人工/LLM 评判回答是否包含正确信息
  - cross_contamination: 回答中是否混入其他领域的知识（不应该有）
  - retrieval_precision: 检索到的记忆是否来自正确领域

对比维度：
  - 同一系统，分别用 vector / topological / hybrid 策略
```

### Benchmark 2: 长期漂移测试 (`benchmarks/long_term_drift.py`)

```
目标：验证系统在领域切换后是否保持旧领域知识。

实验流程：
  Phase A (step 1-100): 只输入 Programming 领域知识
  Phase B (step 101-200): 只输入 Cooking 领域知识
  Phase C (step 201-300): 混合输入
  
  在每个 phase 结束时：
  - 提 10 个 Programming 问题
  - 提 10 个 Cooking 问题
  - 记录 drift_status 和 fingerprint 变化

评估指标：
  - forgetting_rate: Phase B 结束时 Programming 准确率 vs Phase A 结束时
  - drift_detection: 系统是否在 Phase A→B 切换时检测到漂移
  - fingerprint_trajectory: 指纹在三个 phase 中的变化曲线
    （预期：Phase A 稳定 → Phase B 转换 → Phase C 混合新结构）

数据记录：
  每步记录：step, domain, surprise, drift_status, fingerprint (100-dim)
```

### Benchmark 3: 基线对比 (`benchmarks/baseline_comparison.py`)

```
目标：证明 TopoMem 架构的价值——拓扑记忆优于纯向量 RAG。

四个对比系统：

1. TopoMem (topological)
   = 完整系统，retrieval_strategy="topological"

2. TopoMem (hybrid)
   = 完整系统，retrieval_strategy="hybrid"

3. TopoMem (vector-only)
   = 完整系统，retrieval_strategy="vector"
   （相当于关闭拓扑层，只用 ChromaDB 向量检索）

4. Naive RAG
   = 同一 LLM + ChromaDB，无拓扑层、无 adapter、无 self-awareness
   （纯粹的 embedding 检索 + LLM 生成）

5. Pure LLM
   = 同一 LLM，无任何记忆系统
   （baseline: 模型裸跑）

实验流程：
  对 5 个系统各跑一遍 Benchmark 1 + Benchmark 2 的流程
  收集所有指标

指标对比表（期望结果）：
  ┌─────────────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
  │ 指标            │ Pure LLM │ Naive RAG│ TM-Vec   │ TM-Topo  │ TM-Hybrid│
  ├─────────────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
  │ 知识准确率      │ 最低     │ 中       │ 中       │ 高       │ 最高     │
  │ 跨域污染率      │ N/A      │ 中       │ 中       │ 低       │ 低       │
  │ 遗忘率          │ N/A      │ 高       │ 中       │ 低       │ 低       │
  │ 漂移检测        │ N/A      │ N/A      │ N/A      │ 有       │ 有       │
  │ RAM 占用        │ 最低     │ 低       │ 中       │ 中       │ 中       │
  │ 延迟            │ 最低     │ 低       │ 中       │ 中       │ 略高     │
  └─────────────────┴──────────┴──────────┴──────────┴──────────┴──────────┘
```

### Benchmark 4: 资源占用 (`benchmarks/resource_usage.py`)

```
目标：确保系统在硬件约束内运行。

监控方式：
  使用 psutil 库在 process() 前后测量：
  - RSS (Resident Set Size) 内存占用
  - 进程 CPU 时间
  - 操作延迟

采样点：
  在 Benchmark 2 的流程中，每 10 步采样一次

输出表格：
  step | memory_nodes | ram_mb | latency_ms | topo_compute_ms
  ──────────────────────────────────────────────────────────
  10   | 10          | 800    | 5000       | 100
  50   | 50          | 900    | 5500       | 300
  100  | 100         | 1100   | 6000       | 800
  200  | 200         | 1500   | 7000       | 2000
  300  | 300         | 1800   | 8000       | 3500

通过标准：
  - RAM 峰值 < 4 GB
  - 平均延迟 < 60s
  - 拓扑计算 < 10s (at 500 nodes)
```

---

## 4. MVP 通过标准（全部必须满足才算 MVP 成功）

| # | 指标 | 标准 | 验证方式 |
|---|------|------|---------|
| M1 | 系统可运行 | 无崩溃完成 Benchmark 2 全部 300 步 | Benchmark 2 无异常退出 |
| M2 | 拓扑 > 向量（准确率）| TM-Topo accuracy > TM-Vec accuracy | Benchmark 3 |
| M3 | 拓扑 > Naive RAG（一致性）| TM-Topo forgetting < Naive RAG forgetting | Benchmark 3 |
| M4 | 漂移检测有效 | 在 Benchmark 2 Phase A→B 切换时检测到 drifting | Benchmark 2 日志 |
| M5 | RAM 约束 | 峰值 < 4 GB | Benchmark 4 |
| M6 | 延迟约束 | 平均 < 60s 每步 | Benchmark 4 |

### 如果某个标准未通过

| 标准 | 可能原因 | 应对措施 |
|------|---------|---------|
| M1 | OOM、bug | 排查修复（优先） |
| M2 | 拓扑检索不如向量检索 | 分析原因→调整拓扑簇切割阈值/hybrid权重 |
| M3 | 遗忘率高 | 检查 ConsistencyGuard 和 prune 策略 |
| M4 | 漂移未检测到 | 降低 drift_threshold |
| M5 | 内存超标 | 减小 max_nodes / 用更小 embedding 模型 |
| M6 | 太慢 | 减小 max_tokens / 增大 topo_recompute_interval |

---

## 5. 测试语料规格

### 5.1 知识事实格式

```json
{
  "domain": "programming",
  "facts": [
    {
      "id": "prog_001",
      "content": "In Python, list comprehensions are generally faster than equivalent for loops because they are optimized at the bytecode level.",
      "test_question": "Are Python list comprehensions faster or slower than for loops?",
      "expected_keywords": ["faster", "list comprehension", "bytecode"]
    }
  ]
}
```

### 5.2 语料文件位置

```
topomem/data/test_corpus/
├── programming.json    # 20 条编程知识
├── physics.json        # 20 条物理知识
├── history.json        # 20 条历史知识
├── cooking.json        # 20 条烹饪知识
└── geography.json      # 20 条地理知识
```

### 5.3 语料准备方式

由执行 agent 创建，或由 LLM 辅助生成。要求：
- 事实必须是可验证的
- 问题的答案必须唯一存在于对应领域的事实中
- 跨领域问题应尽量避免歧义

---

## 6. 集成测试清单

```python
# test_integration.py

def test_system_init():
    """系统应成功初始化所有组件"""

def test_single_process():
    """单次 process() 应返回 ProcessResult"""

def test_process_with_memory():
    """先存入知识，再提问，应能引用知识回答"""

def test_process_10_steps():
    """连续 10 步 process 应无报错"""

def test_adapter_auto_creation():
    """输入足够多的新领域知识后应自动创建 adapter"""

def test_drift_detection_on_domain_switch():
    """领域切换时应检测到漂移"""

def test_save_and_load():
    """save 后 load，系统行为应一致"""

def test_system_status():
    """get_status() 应返回完整的状态信息"""

def test_process_result_latency():
    """单次 process 延迟应在合理范围内"""

def test_memory_prune_at_capacity():
    """达到容量上限时应自动 prune"""
```

---

## 7. 报告输出格式

每次 benchmark 运行后生成报告：

```
topomem/results/
├── benchmark_YYYYMMDD_HHMMSS/
│   ├── report.md              # 人可读的实验报告
│   ├── config.json            # 使用的配置
│   ├── knowledge_consistency.json
│   ├── long_term_drift.json
│   ├── baseline_comparison.json
│   ├── resource_usage.json
│   └── fingerprint_trajectory.png  # 指纹变化曲线图（如果可生成）
```
