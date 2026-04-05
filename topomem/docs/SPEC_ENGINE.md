# SPEC_ENGINE.md — 推理引擎集成技术规格

> **对应模块**：`topomem/engine.py` + `topomem/embedding.py`  
> **对应阶段**：Phase 3（embedding 部分属 Phase 1）  
> **前置依赖**：sentence-transformers, llama-cpp-python（或 transformers fallback）  
> **被依赖方**：system.py, adapters.py

---

## 1. 模块职责

推理引擎层包含两个模块：

| 模块 | 核心角色 |
|------|---------|
| `EmbeddingManager` | **感知器** — 将原始文本转为高维特征向量 |
| `ReasoningEngine` | **思考器** — 接收 prompt + 记忆上下文 → 生成回答 |

**关键原则**：引擎层**不持有知识**。它是一个纯粹的计算单元，所有知识来自 Layer 2 的记忆系统。

---

## 2. EmbeddingManager 规格

```python
class EmbeddingManager:
    """文本 → 特征向量的转换器。
    
    当前使用 sentence-transformers 的 all-MiniLM-L6-v2。
    该模型专为语义相似度任务训练，384 维输出。
    """
    
    def __init__(self, config: EmbeddingConfig):
        """
        初始化：
        1. 加载 SentenceTransformer(config.model_name)
        2. 设置 device=config.device（默认 "cpu"）
        3. 验证输出维度 == config.dimension (384)
        
        首次加载会从 HuggingFace 下载模型（~90MB）。
        后续使用缓存。
        
        注意：sentence-transformers 依赖 torch。
        在 CPU-only 环境下，torch 会使用 CPU 后端。
        """
    
    def encode(self, text: str) -> np.ndarray:
        """单条文本 → 384 维向量。
        
        实现：
        1. model.encode(text, convert_to_numpy=True)
        2. 返回 shape (384,), dtype float32
        
        性能：单条文本 < 50ms (CPU)
        """
    
    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """批量编码。
        
        实现：
        1. model.encode(texts, batch_size=config.batch_size, 
                        convert_to_numpy=True, show_progress_bar=False)
        2. 返回 shape (N, 384)
        
        性能：32 条文本 < 500ms (CPU)
        """
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """余弦相似度。
        
        实现：dot(a, b) / (norm(a) * norm(b))
        返回范围 [-1, 1]，越大越相似。
        """
```

---

## 3. ReasoningEngine 规格

```python
class ReasoningEngine:
    """轻量 LLM 推理引擎。
    
    主选后端：llama-cpp-python（加载 GGUF 量化模型）
    备选后端：transformers（加载 HuggingFace 模型）
    
    后端选择逻辑：
    1. 尝试 import llama_cpp
    2. 检查 config.model_path 是否存在（GGUF 文件）
    3. 如果都满足 → 使用 llama-cpp 后端
    4. 否则 → 使用 transformers 后端（config.fallback_model_name）
    5. config.use_fallback = True 强制使用 transformers
    """
    
    def __init__(self, config: EngineConfig):
        """
        llama-cpp 后端初始化：
        1. from llama_cpp import Llama
        2. self._model = Llama(
               model_path=config.model_path,
               n_ctx=config.n_ctx,        # 2048
               n_threads=config.n_threads, # 4
               verbose=False
           )
        
        transformers 后端初始化：
        1. from transformers import AutoModelForCausalLM, AutoTokenizer
        2. self._tokenizer = AutoTokenizer.from_pretrained(config.fallback_model_name)
        3. self._model = AutoModelForCausalLM.from_pretrained(
               config.fallback_model_name,
               torch_dtype=torch.float32,  # CPU 不支持 float16
               device_map="cpu"
           )
        
        内存预算：
        - GGUF Q4: ~400MB
        - transformers float32: ~2GB（0.5B 模型）
        """
    
    def generate(
        self,
        prompt: str,
        context: Optional[List[MemoryNode]] = None,
        adapter: Optional[BaseAdapter] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """核心推理方法。
        
        步骤：
        1. 构造完整 prompt（见 §4 prompt 模板）
           a. system_prompt = adapter.system_prompt（如果有 adapter）
                              or DEFAULT_SYSTEM_PROMPT
           b. context_block = 格式化记忆上下文（见 §4.2）
           c. full_prompt = 组装 system + context + user query
        
        2. 调用模型生成
           llama-cpp:
             result = self._model.create_chat_completion(
                 messages=[
                     {"role": "system", "content": system_prompt},
                     {"role": "user", "content": context_block + "\n\n" + prompt}
                 ],
                 max_tokens=max_tokens or config.max_tokens,
                 temperature=temperature or config.temperature
             )
             return result['choices'][0]['message']['content']
           
           transformers:
             inputs = self._tokenizer(full_prompt, return_tensors="pt")
             outputs = self._model.generate(**inputs, max_new_tokens=max_tokens)
             return self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        3. 返回生成的文本
        
        性能约束：
        - GGUF Q4 CPU: ~5-15 tok/s → 256 tokens 约 17-50s
        - transformers float32 CPU: ~2-5 tok/s → 256 tokens 约 50-120s
        """
    
    def estimate_tokens(self, text: str) -> int:
        """粗略估算文本的 token 数（用于上下文窗口管理）。
        
        简单实现：len(text) / 4（英文）或 len(text) / 2（中文）
        精确实现（如有 tokenizer）：len(tokenizer.encode(text))
        """
```

---

## 4. Prompt 模板设计

### 4.1 系统提示

```python
DEFAULT_SYSTEM_PROMPT = """You are a precise and consistent reasoning assistant. 
You answer questions based on the provided context. 
If the context doesn't contain relevant information, say so clearly.
Be concise and factual."""
```

Adapter 的 system_prompt 会替换这个默认提示（见 SPEC_ADAPTERS.md）。

### 4.2 记忆上下文格式化

```python
def format_memory_context(memories: List[MemoryNode]) -> str:
    """将检索到的记忆格式化为 LLM 可理解的上下文。
    
    格式：
    
    --- Relevant Knowledge ---
    [1] {memory.content}
    (Relevance: {cluster_info}, Accessed: {access_count} times)
    
    [2] {memory.content}
    (Relevance: {cluster_info}, Accessed: {access_count} times)
    
    ...
    --- End of Knowledge ---
    
    设计理由：
    - 编号方便模型引用
    - cluster_info 帮助模型理解知识的组织方式
    - access_count 暗示知识的可靠性
    - 使用清晰的分隔符防止上下文和用户输入混淆
    """
```

### 4.3 完整 prompt 组装

```
[System]
{adapter.system_prompt or DEFAULT_SYSTEM_PROMPT}

[User]
--- Relevant Knowledge ---
[1] Python's GIL prevents true multi-threading for CPU-bound tasks.
(Cluster: Programming/Python, Accessed: 12 times)

[2] asyncio provides cooperative multitasking through coroutines.
(Cluster: Programming/Python, Accessed: 8 times)
--- End of Knowledge ---

Based on the above knowledge, answer the following question:
{user_query}
```

### 4.4 上下文窗口管理

```python
def truncate_context(
    memories: List[MemoryNode],
    max_context_tokens: int = 1024  # 保留一半窗口给生成
) -> List[MemoryNode]:
    """如果记忆太多，截断到 token 预算内。
    
    策略：
    1. 从最相关的开始累积 token 数
    2. 超过预算则截断
    3. 保证至少保留 1 条记忆
    
    context 窗口分配（n_ctx=2048）：
    - system prompt: ~100 tokens
    - memory context: ≤ 1024 tokens
    - user query: ~200 tokens
    - generation: ≤ 256 tokens
    - 安全余量: ~468 tokens
    """
```

---

## 5. 知识提取（从回答中提取可存储的新知识）

```python
def extract_knowledge(
    user_query: str,
    response: str,
    engine: ReasoningEngine
) -> Optional[str]:
    """从模型回答中提取值得存储的新知识。
    
    MVP 实现（简单规则）：
    1. 如果 response 长度 < 20 字符 → None（太短，无知识价值）
    2. 如果 response 包含 "I don't know" / "not sure" → None
    3. 否则，合并 query + response 的关键信息作为新记忆内容：
       knowledge = f"Q: {query}\nA: {response}"
    
    未来增强（可选）：
    - 用 LLM 自身判断回答中是否有新知识
    - 用 NLI 模型检查是否与已有知识矛盾
    """
```

---

## 6. 模型文件管理

### 6.1 GGUF 模型获取

```
模型：Qwen2.5-0.5B-Instruct-Q4_K_M.gguf
来源：HuggingFace (Qwen/Qwen2.5-0.5B-Instruct-GGUF)
大小：~400MB
存放路径：topomem/data/models/qwen2.5-0.5b-instruct-q4_k_m.gguf

下载方式（执行 agent 操作）：
  方法 1: huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct-GGUF \
           qwen2.5-0.5b-instruct-q4_k_m.gguf \
           --local-dir topomem/data/models/
  
  方法 2: 直接 wget/curl 下载
  
  方法 3: 用 Python huggingface_hub.hf_hub_download()
```

### 6.2 Fallback 模型

```
如果 GGUF 不可用，transformers 后端自动从 HuggingFace 下载：
  Qwen/Qwen2.5-0.5B-Instruct
  约 1GB 下载，2GB 加载到 RAM（float32）
```

---

## 7. 错误处理

| 场景 | 处理方式 |
|------|---------|
| 模型文件不存在 | 自动切换到 transformers fallback |
| llama-cpp import 失败 | 自动切换到 transformers fallback |
| 生成过程 OOM | 捕获异常，减少 max_tokens 重试（最低 64） |
| 生成超时（> 120s） | 中断生成，返回已有部分 |
| 空 prompt | 抛出 ValueError |
| context 为空列表 | 正常生成（不注入上下文）|
| embedding 模型加载失败 | 抛出 RuntimeError（无法 fallback，这是必须组件）|

---

## 8. 单元测试清单

```python
# test_embedding.py
def test_encode_returns_correct_shape():
    """encode 应返回 (384,) ndarray"""

def test_encode_batch_shapes():
    """encode_batch 5 条文本应返回 (5, 384)"""

def test_similarity_same_text():
    """同一文本的 similarity 应接近 1.0"""

def test_similarity_different_text():
    """不相关文本的 similarity 应较低"""

def test_encode_empty_string():
    """空字符串应能编码（不报错）"""

def test_encode_chinese():
    """中文文本应能编码"""

# test_engine.py
def test_generate_basic():
    """基本生成应返回非空字符串"""

def test_generate_with_context():
    """带 context 的生成应引用上下文内容"""

def test_generate_respects_max_tokens():
    """输出 token 数不应超过 max_tokens"""

def test_format_memory_context():
    """上下文格式化应包含所有记忆内容"""

def test_truncate_context():
    """超长上下文应被正确截断"""

def test_fallback_engine():
    """主引擎不可用时应 fallback 到备选"""

def test_estimate_tokens():
    """token 估算应在合理范围内"""
```
