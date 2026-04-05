# SPEC_ADAPTERS.md — 动态塑造机制技术规格

> **对应模块**：`topomem/adapters.py`  
> **对应阶段**：Phase 4（在本项目中排在 Phase 5/自我认知之后）  
> **前置依赖**：memory.py, embedding.py, self_awareness.py  
> **被依赖方**：system.py

---

## 1. 设计哲学

### 为什么需要动态塑造

小模型（0.5B）的一个根本问题：**用同一组参数处理所有领域是低效的**。

人类在不同场景中会切换"思维模式"——写代码时你是逻辑模式，聊天时你是社交模式，做数学时你是抽象模式。这些模式不是重新学习，而是**激活不同的先验**。

TopoMem 的动态塑造做同样的事：根据当前 query 所属的知识域（由拓扑簇决定），选择不同的行为模式（Adapter）。

### MVP vs 未来

| 维度 | MVP（Prompt Adapter） | 未来（LoRA Adapter） |
|------|----------------------|---------------------|
| 行为修改方式 | 改变 system prompt | 修改模型权重 |
| 参数开销 | 0 额外参数 | ~1-4MB / adapter |
| 效果深度 | 浅层（指令层面） | 深层（特征层面） |
| 需要 GPU | 否 | 是 |
| 创建方式 | 规则生成 | 微调训练 |

**关键设计**：两者共享同一个 `BaseAdapter` 抽象接口，切换时只需要替换实现。

---

## 2. 抽象接口（为 LoRA 预留）

```python
from abc import ABC, abstractmethod

class BaseAdapter(ABC):
    """Adapter 抽象基类。
    
    所有 adapter（无论 Prompt 还是 LoRA）必须实现这两个方法。
    这保证了 AdapterPool 和 system.py 不需要知道底层实现。
    
    未来切换到 LoRA 时：
    1. 新建 LoRAAdapter(BaseAdapter) 类
    2. 修改 config.adapter_backend = "lora"
    3. AdapterPool 根据 backend 创建不同类型的 adapter
    4. system.py 完全不需要改动
    """
    
    @abstractmethod
    def apply(self, engine: ReasoningEngine, prompt: str) -> str:
        """将 adapter 的效果应用到推理过程中。
        
        Prompt 实现：修改 system prompt → 返回修改后的完整 prompt
        LoRA 实现：加载 LoRA 权重 → 返回原始 prompt（权重变化已生效）
        """
    
    @abstractmethod
    def get_domain_embedding(self) -> np.ndarray:
        """返回此 adapter 所代表的领域的 embedding。
        
        用于匹配 query 和 adapter 之间的相关性。
        """
    
    @property
    @abstractmethod
    def adapter_id(self) -> str: ...
    
    @property
    @abstractmethod
    def adapter_type(self) -> str: ...  # "prompt" or "lora"
```

---

## 3. PromptAdapter 实现

```python
@dataclass
class PromptAdapter(BaseAdapter):
    """基于 system prompt 的行为适配器（MVP 实现）。"""
    
    id: str                           # UUID
    name: str                         # 人可读名称（如 "Python 编程"）
    system_prompt: str                # 定制化系统提示
    domain_keywords: List[str]        # 关联的领域关键词
    domain_embedding: np.ndarray      # 领域中心 embedding（384 维）
    topological_cluster: int          # 对应的拓扑簇 ID
    
    # 生命周期统计
    created_at: float
    usage_count: int = 0
    effectiveness_score: float = 0.5  # 初始中性分数
    last_used: float = 0.0
    
    def apply(self, engine: ReasoningEngine, prompt: str) -> str:
        """注入定制化 system prompt。
        
        实现：
        返回的不是修改后的 prompt，而是让 engine.generate() 
        使用 self.system_prompt 替代 DEFAULT_SYSTEM_PROMPT。
        
        实际上 apply 的作用是返回 system_prompt，
        由 engine.generate() 在构造 chat messages 时使用。
        """
        self.usage_count += 1
        self.last_used = time.time()
        return self.system_prompt
    
    def get_domain_embedding(self) -> np.ndarray:
        return self.domain_embedding
```

### 3.1 System Prompt 模板

```python
ADAPTER_PROMPT_TEMPLATE = """You are a specialized assistant for {domain_name}.

Your expertise includes: {domain_keywords}.

When answering questions in this domain:
- Prioritize accuracy and domain-specific terminology
- Reference relevant context from your knowledge base
- Be precise and technical when appropriate
- If unsure, clearly state the limitation

{custom_instructions}"""
```

不同领域的 adapter 示例：

```python
# 编程领域
PromptAdapter(
    name="Programming",
    system_prompt="""You are a specialized assistant for Programming.
Your expertise includes: algorithms, data structures, Python, debugging.
When answering questions in this domain:
- Provide code examples when helpful
- Explain time/space complexity
- Suggest best practices
- Reference relevant documentation""",
    domain_keywords=["code", "programming", "algorithm", "function", "debug"]
)

# 科学知识领域
PromptAdapter(
    name="Science",
    system_prompt="""You are a specialized assistant for Science.
Your expertise includes: physics, chemistry, biology, mathematics.
When answering questions in this domain:
- Use precise scientific terminology
- Reference known laws and theories
- Distinguish between established facts and hypotheses
- Express quantities with appropriate units""",
    domain_keywords=["science", "physics", "chemistry", "formula", "theory"]
)
```

---

## 4. AdapterPool 完整 API

```python
class AdapterPool:
    """管理 Adapter 的创建、选择、进化和淘汰。
    
    AdapterPool 是动态塑造的核心：它根据 query 的拓扑位置
    自动选择最匹配的 adapter，并随使用经验进化。
    """
    
    def __init__(self, config: AdapterConfig, embedding_mgr: EmbeddingManager):
        """
        初始化：
        - self._adapters: Dict[str, BaseAdapter] = {}
        - self._default_adapter: BaseAdapter  ← 使用 DEFAULT_SYSTEM_PROMPT 的通用 adapter
        - config.adapter_backend 决定创建 PromptAdapter 还是 LoRAAdapter
        """
    
    def select_adapter(
        self,
        query_embedding: np.ndarray,
        memory_graph: MemoryGraph
    ) -> Tuple[BaseAdapter, float]:
        """根据 query 的拓扑位置选择最合适的 adapter。
        
        选择算法：
        
        1. 如果没有任何非 default 的 adapter → 返回 (default_adapter, 0.0)
        
        2. 计算 query_embedding 与每个 adapter.domain_embedding 的余弦相似度
        
        3. 选择相似度最高的 adapter，记为 best_adapter, best_sim
        
        4. 置信度门槛：
           如果 best_sim < 0.3 → 返回 (default_adapter, best_sim)
           （query 与所有 adapter 都不够匹配，用通用 adapter 更安全）
        
        5. 返回 (best_adapter, best_sim)
        
        第二返回值 (surprise_score) 的含义：
        - surprise = 1.0 - best_sim
        - 越高 → query 越"出乎意料"（不属于任何已知领域）
        - 用于触发新 adapter 的创建
        """
    
    def create_adapter(
        self,
        cluster_id: int,
        representative_memories: List[MemoryNode],
        engine: Optional[ReasoningEngine] = None
    ) -> BaseAdapter:
        """从拓扑簇中自动生成新 adapter。
        
        步骤：
        
        1. 领域 embedding：
           domain_embedding = mean(m.embedding for m in representative_memories)
        
        2. 领域关键词提取：
           方法 A（有 engine）：让 LLM 从记忆内容中提取关键词
             prompt = "Extract 5-8 domain keywords from these texts: ..."
           方法 B（无 engine）：简单 TF-IDF 或词频统计
        
        3. 领域名称：
           方法 A：让 LLM 命名
           方法 B：取最高频关键词
        
        4. System Prompt 生成：
           用 ADAPTER_PROMPT_TEMPLATE 填充 domain_name 和 keywords
        
        5. 创建 PromptAdapter 并加入 pool
        
        6. 如果 len(self._adapters) > config.max_adapters:
           self._prune_adapters()
        
        返回新创建的 adapter。
        """
    
    def evolve_adapter(
        self,
        adapter_id: str,
        feedback: float
    ) -> None:
        """根据反馈调整 adapter 的效果评分。
        
        更新规则：
        new_score = old_score * config.effectiveness_decay + feedback * (1 - decay)
        
        其中 feedback ∈ [0, 1]：
        - 1.0 = 用户对这次回答非常满意
        - 0.5 = 中性（默认，无明确反馈）
        - 0.0 = 回答很差
        
        MVP 阶段：由于没有用户反馈机制，默认 feedback=0.5。
        未来可以接入：
        - 用户显式评分
        - 基于 LLM 自评
        - 基于回答是否被后续对话引用
        """
    
    def _prune_adapters(self) -> List[str]:
        """淘汰低效 adapter。
        
        淘汰策略：
        1. 保留 default_adapter（不可淘汰）
        2. 按 effectiveness_score * log(1 + usage_count) 排序
        3. 如果 usage_count < config.min_usage_for_keep → 优先淘汰
        4. 移除得分最低的，直到 adapter 数 <= config.max_adapters
        
        返回被移除的 adapter_id 列表。
        """
    
    def get_all_adapters(self) -> List[BaseAdapter]:
        """返回所有 adapter（包括 default）。"""
    
    # =========== 序列化 ===========
    
    def save(self, path: str) -> None:
        """保存 adapter pool 到磁盘。"""
    
    def load(self, path: str) -> None:
        """从磁盘加载。"""
```

---

## 5. Surprise / Tension 信号系统

从 unified-sel 的 Structure Pool 迁移而来的信号机制：

### 5.1 Surprise 信号

```python
def compute_surprise(
    query_embedding: np.ndarray,
    adapter_pool: AdapterPool
) -> float:
    """query 的"意外度"。
    
    surprise = 1.0 - max_similarity_to_any_adapter
    
    含义：
    - surprise ≈ 0 → query 完全在已知领域内
    - surprise ≈ 0.5 → query 与已知领域有一定关系
    - surprise ≈ 1.0 → query 完全是新领域
    
    用途：
    - surprise > 0.7 → 触发新 adapter 创建（如果该簇有足够记忆）
    """
```

### 5.2 Tension 信号

```python
def compute_tension(
    self_awareness: SelfAwareness,
    window: int = 5
) -> float:
    """系统的"张力"= 最近 N 次拓扑指纹变化的平均速率。
    
    tension = mean(wasserstein_distance(fp[i], fp[i-1]) for i in last N)
    
    含义：
    - tension ≈ 0 → 系统稳定
    - tension 中等 → 正常学习演化
    - tension 高 → 知识快速变化（可能需要整理）
    
    用途：
    - tension > drift_threshold → 触发记忆整理（而非创建新 adapter）
    """
```

### 5.3 决策矩阵

```
┌─────────────────────┬──────────────────────┬──────────────────────┐
│                     │ Tension 低           │ Tension 高           │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Surprise 低         │ 使用现有 adapter     │ 触发记忆整理         │
│ (query 在已知域)    │ （常规操作）         │ （知识在变化，       │
│                     │                      │  先整理再继续）      │
├─────────────────────┼──────────────────────┼──────────────────────┤
│ Surprise 高         │ 创建新 adapter       │ 触发记忆整理 +       │
│ (query 在新域)      │ （新领域出现，       │ 暂缓 adapter 创建    │
│                     │  为它建立专属模式）   │ （系统不稳定时       │
│                     │                      │  先稳定再扩展）      │
└─────────────────────┴──────────────────────┴──────────────────────┘

阈值参考：
- surprise_threshold: 0.7（超过此值视为新领域）
- tension_threshold: config.drift_threshold（与漂移检测共用）
```

---

## 6. Adapter 生命周期

```
            ┌──────────┐
            │  不存在   │
            └────┬─────┘
                 │ surprise > 0.7 且该簇有 ≥ 5 条记忆
                 ▼
            ┌──────────┐
            │  新建     │ usage_count = 0, effectiveness = 0.5
            └────┬─────┘
                 │ 被 select_adapter 选中
                 ▼
            ┌──────────┐
            │  活跃     │ usage_count 递增, effectiveness 更新
            └────┬─────┘
                 │ usage_count < min_usage 且 pool 满
                 │ 或 effectiveness 持续低于 0.3
                 ▼
            ┌──────────┐
            │  淘汰     │ 从 pool 中移除
            └──────────┘
```

---

## 7. 单元测试清单

```python
# test_adapters.py

# === BaseAdapter 接口测试 ===

def test_prompt_adapter_implements_base():
    """PromptAdapter 应正确实现 BaseAdapter 接口"""

def test_adapter_apply_returns_system_prompt():
    """apply() 应返回 system_prompt 字符串"""

def test_adapter_domain_embedding_shape():
    """get_domain_embedding() 应返回 (384,) 向量"""

# === AdapterPool 测试 ===

def test_select_default_when_empty():
    """无自定义 adapter 时应返回 default"""

def test_select_best_match():
    """应选择与 query 最匹配的 adapter"""

def test_select_fallback_low_similarity():
    """similarity < 0.3 时应 fallback 到 default"""

def test_create_adapter_from_cluster():
    """应成功从记忆簇创建新 adapter"""

def test_create_adapter_auto_prune():
    """超过 max_adapters 时应自动淘汰"""

def test_evolve_adapter_updates_score():
    """feedback 应更新 effectiveness_score"""

def test_prune_removes_lowest():
    """prune 应移除 effectiveness 最低的"""

def test_prune_keeps_default():
    """default adapter 不应被 prune"""

# === Surprise/Tension 测试 ===

def test_surprise_known_domain():
    """已知领域的 query → surprise 应较低"""

def test_surprise_unknown_domain():
    """全新领域的 query → surprise 应较高"""

def test_tension_stable_system():
    """稳定系统的 tension 应接近 0"""

def test_tension_changing_system():
    """快速变化的系统 tension 应较高"""

# === 序列化测试 ===

def test_adapter_pool_save_load():
    """save 后 load，所有 adapter 应完整恢复"""
```
