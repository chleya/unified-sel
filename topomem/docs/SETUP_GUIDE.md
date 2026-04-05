# SETUP_GUIDE.md — 环境搭建与依赖安装指南

> **面向**：执行 agent  
> **目标**：从零搭建 TopoMem Reasoner v0.1 的开发环境  
> **前提**：操作系统 Windows，Python 3.14，无独立 GPU

---

## 1. 关键约束

| 约束 | 说明 |
|------|------|
| **C 盘空间不足** | 所有依赖和数据必须安装到 **F 盘**（项目目录下） |
| **无独立 GPU** | 所有计算基于 CPU；torch 使用 CPU 后端 |
| **RAM 16GB** | 注意内存占用，避免同时加载多个大模型 |
| **Python 3.14** | 较新版本，部分库可能需要兼容性检查 |

---

## 2. 虚拟环境

虚拟环境已创建在 `F:\unified-sel\.venv\`。

**激活方式**：
```powershell
# PowerShell
F:\unified-sel\.venv\Scripts\Activate.ps1

# 或直接使用完整路径调用 python/pip
F:\unified-sel\.venv\Scripts\python.exe
F:\unified-sel\.venv\Scripts\pip.exe
```

**所有 pip install 和 python 命令都必须使用 venv 中的可执行文件。**

---

## 3. 依赖安装

### 3.1 已确认安装的包

| 包 | 版本 | 状态 |
|---|------|------|
| numpy | 2.4.4 | ✅ 已装 |
| scipy | (随 scikit-learn 安装) | ✅ 已装 |
| scikit-learn | (已装) | ✅ 已装 |
| networkx | 3.6.1 | ✅ 已装 |
| gudhi | 3.12.0 | ✅ 已装 |
| ripser | 0.6.14 | ✅ 已装 |
| persim | 0.3.8 | ✅ 已装 |
| chromadb | 1.5.5 | ✅ 已装 |
| Cython | 3.2.4 | ✅ 已装 |

### 3.2 待安装的包

按此顺序逐个安装（避免超时，每个命令单独执行）：

```powershell
# 1. sentence-transformers（embedding 模型）
F:\unified-sel\.venv\Scripts\pip.exe install sentence-transformers

# 2. llama-cpp-python（CPU 推理引擎）
# Windows CPU-only 安装：
F:\unified-sel\.venv\Scripts\pip.exe install llama-cpp-python

# 如果上面编译失败，尝试预编译 wheel：
F:\unified-sel\.venv\Scripts\pip.exe install llama-cpp-python --prefer-binary

# 如果仍然失败，改用 transformers 作为 fallback：
F:\unified-sel\.venv\Scripts\pip.exe install transformers accelerate

# 3. psutil（资源监控，benchmark 需要）
F:\unified-sel\.venv\Scripts\pip.exe install psutil

# 4. pytest（测试框架）
F:\unified-sel\.venv\Scripts\pip.exe install pytest
```

### 3.3 安装验证脚本

安装完成后运行以下验证：

```python
# 保存为 topomem/tests/test_infra.py 并运行
"""基础设施冒烟测试。"""

import sys
import numpy as np

def test_python_version():
    assert sys.version_info >= (3, 10), f"Need Python 3.10+, got {sys.version}"
    print(f"✓ Python {sys.version}")

def test_numpy():
    a = np.random.rand(10, 384)
    assert a.shape == (10, 384)
    print(f"✓ numpy {np.__version__}")

def test_scipy():
    from scipy.spatial.distance import pdist, squareform
    pts = np.random.rand(10, 5)
    D = squareform(pdist(pts))
    assert D.shape == (10, 10)
    print("✓ scipy (pdist OK)")

def test_networkx():
    import networkx as nx
    G = nx.Graph()
    G.add_node("a", data=123)
    G.add_node("b", data=456)
    G.add_edge("a", "b")
    assert G.number_of_nodes() == 2
    print(f"✓ networkx {nx.__version__}")

def test_ripser():
    from ripser import ripser
    pts = np.random.rand(20, 5)
    result = ripser(pts, maxdim=1)
    assert len(result['dgms']) == 2  # H0 and H1
    print(f"✓ ripser (H0: {len(result['dgms'][0])} features, H1: {len(result['dgms'][1])} features)")

def test_persim():
    from ripser import ripser
    from persim import wasserstein
    pts = np.random.rand(20, 5)
    result = ripser(pts, maxdim=0)
    d = wasserstein(result['dgms'][0], result['dgms'][0])
    assert d == 0.0 or abs(d) < 1e-10
    print(f"✓ persim (self-distance: {d})")

def test_gudhi():
    import gudhi
    rips = gudhi.RipsComplex(points=np.random.rand(10, 3).tolist(), max_edge_length=2.0)
    st = rips.create_simplex_tree(max_dimension=2)
    st.compute_persistence()
    print(f"✓ gudhi {gudhi.__version__}")

def test_chromadb():
    import chromadb
    client = chromadb.Client()  # ephemeral, in-memory
    col = client.create_collection("test_smoke")
    col.add(ids=["1"], embeddings=[[0.1]*384], documents=["hello"])
    results = col.query(query_embeddings=[[0.1]*384], n_results=1)
    assert results['ids'][0][0] == "1"
    client.delete_collection("test_smoke")
    print(f"✓ chromadb {chromadb.__version__}")

def test_sentence_transformers():
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    vec = model.encode("hello world")
    assert vec.shape == (384,)
    print(f"✓ sentence-transformers (output dim: {vec.shape[0]})")

def test_llm_backend():
    """测试 LLM 推理后端（llama-cpp 或 transformers fallback）"""
    try:
        from llama_cpp import Llama
        print("✓ llama-cpp-python available")
        return
    except ImportError:
        print("  llama-cpp-python not available, checking transformers fallback...")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("✓ transformers available (will use as fallback)")
    except ImportError:
        print("✗ NEITHER llama-cpp-python NOR transformers available!")
        raise ImportError("No LLM backend available")

if __name__ == "__main__":
    tests = [
        test_python_version, test_numpy, test_scipy, test_networkx,
        test_ripser, test_persim, test_gudhi, test_chromadb,
        test_sentence_transformers, test_llm_backend,
    ]
    passed = 0
    failed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"✗ {t.__name__}: {e}")
            failed += 1
    print(f"\n{'='*40}")
    print(f"Results: {passed} passed, {failed} failed")
    if failed > 0:
        print("Fix failed tests before proceeding to Phase 1.")
    else:
        print("All infrastructure tests passed! Ready for Phase 1.")
```

---

## 4. 模型下载

### 4.1 Qwen2.5-0.5B GGUF（主选推理引擎）

```powershell
# 方法 1: 用 huggingface-cli（需先 pip install huggingface-hub）
F:\unified-sel\.venv\Scripts\pip.exe install huggingface-hub

F:\unified-sel\.venv\Scripts\python.exe -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='Qwen/Qwen2.5-0.5B-Instruct-GGUF',
    filename='qwen2.5-0.5b-instruct-q4_k_m.gguf',
    local_dir='F:/unified-sel/topomem/data/models'
)
print('Download complete')
"

# 方法 2: 直接 curl 下载（如果 huggingface-cli 不可用）
# 从 HuggingFace 网页手动下载到 F:\unified-sel\topomem\data\models\
```

**预期文件**：`F:\unified-sel\topomem\data\models\qwen2.5-0.5b-instruct-q4_k_m.gguf`  
**预期大小**：~400MB

### 4.2 Embedding 模型

sentence-transformers 会自动下载 all-MiniLM-L6-v2 到 HuggingFace 缓存。

**注意**：默认缓存在 C 盘 `~/.cache/huggingface/`。如果 C 盘空间不足，设置环境变量：
```powershell
$env:HF_HOME = "F:\unified-sel\.cache\huggingface"
$env:TRANSFORMERS_CACHE = "F:\unified-sel\.cache\huggingface"
```

或在 Python 中：
```python
import os
os.environ["HF_HOME"] = r"F:\unified-sel\.cache\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"F:\unified-sel\.cache\huggingface"
```

---

## 5. 项目结构验证

安装完成后，项目目录应如下：

```
F:\unified-sel\
├── .venv/                              # Python 虚拟环境
│   ├── Scripts/python.exe
│   ├── Scripts/pip.exe
│   └── Lib/site-packages/             # 所有依赖包
├── core/                              # 原 unified-sel 代码
├── topomem/
│   ├── __init__.py                    ✅ 已创建
│   ├── config.py                      ✅ 已创建
│   ├── docs/
│   │   ├── ARCHITECTURE.md            ✅ 已创建
│   │   ├── SETUP_GUIDE.md             ✅ 本文件
│   │   ├── SPEC_TOPOLOGY.md           ✅ 已创建
│   │   ├── SPEC_MEMORY.md             ✅ 已创建
│   │   ├── SPEC_ENGINE.md             ✅ 已创建
│   │   ├── SPEC_SELF_AWARENESS.md     ✅ 已创建
│   │   ├── SPEC_ADAPTERS.md           ✅ 已创建
│   │   └── SPEC_INTEGRATION.md        ✅ 已创建
│   ├── tests/
│   │   ├── __init__.py                ✅ 已创建
│   │   └── test_infra.py              ← 待创建（见上方验证脚本）
│   ├── benchmarks/
│   │   └── __init__.py                ✅ 已创建
│   ├── data/
│   │   ├── models/
│   │   │   └── qwen2.5-0.5b-instruct-q4_k_m.gguf  ← 待下载
│   │   └── test_corpus/               ← 待创建测试语料
│   └── results/
├── TOPOMEM_PLAN.md                    ✅ 已创建
└── TOPOMEM_STATUS.md                  ← 待创建
```

---

## 6. 执行检查清单

执行 agent 请按以下顺序完成，每步完成后打 ✓：

```
Phase 0 检查清单：

[ ] 1. 安装 sentence-transformers 到 venv
[ ] 2. 安装 llama-cpp-python 到 venv（或 transformers fallback）
[ ] 3. 安装 psutil, pytest 到 venv
[ ] 4. 设置 HF_HOME 环境变量指向 F 盘
[ ] 5. 创建 test_infra.py（复制本文件中的验证脚本）
[ ] 6. 运行 test_infra.py，确保全部通过
[ ] 7. 下载 Qwen2.5-0.5B GGUF 模型文件
[ ] 8. 验证模型文件可加载（如果 llama-cpp 可用）
[ ] 9. 在 TOPOMEM_STATUS.md 中记录 Phase 0 完成
```
