# TopoMem 磁盘存储策略

## 重要原则

**所有模型、缓存、数据文件必须存储在 F 盘，严禁占用 C 盘空间。**

## 环境变量配置

以下环境变量必须在导入任何 HuggingFace 库之前设置：

```python
import os
os.environ["HF_HOME"] = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ["TRANSFORMERS_CACHE"] = r"F:\unified-sel\topomem\data\models\hf_cache"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"F:\unified-sel\topomem\data\models\hf_cache"
```

这些变量已永久设置到系统环境（通过 `setx`），但当前 Python 进程仍需显式设置。

## 目录结构

```
F:\unified-sel\topomem\data\
├── models\                              # 所有模型文件
│   ├── hf_cache\                        # HuggingFace 缓存目录
│   │   └── hub\                         # 实际缓存位置
│   │       └── models--Qwen--Qwen2.5-0.5B-Instruct\
│   ├── *.gguf                           # GGUF 模型文件（已下载 8 个）
│   └── .cache\                          # 下载缓存
├── chromadb\                            # ChromaDB 向量数据库
└── test_corpus\                         # 测试语料
```

## 已下载模型

| 模型 | 大小 | 位置 |
|------|------|------|
| qwen2.5-0.5b-instruct-fp16.gguf | 1.27 GB | F:\unified-sel\topomem\data\models\ |
| qwen2.5-0.5b-instruct-q2_k.gguf | 415 MB | F:\unified-sel\topomem\data\models\ |
| qwen2.5-0.5b-instruct-q3_k_m.gguf | 432 MB | F:\unified-sel\topomem\data\models\ |
| qwen2.5-0.5b-instruct-q4_0.gguf | 429 MB | F:\unified-sel\topomem\data\models\ |
| qwen2.5-0.5b-instruct-q4_k_m.gguf | 491 MB | F:\unified-sel\topomem\data\models\ |
| qwen2.5-0.5b-instruct-q5_0.gguf | 490 MB | F:\unified-sel\topomem\data\models\ |
| qwen2.5-0.5b-instruct-q6_k.gguf | 650 MB | F:\unified-sel\topomem\data\models\ |
| qwen2.5-0.5b-instruct-q8_0.gguf | 676 MB | F:\unified-sel\topomem\data\models\ |
| Qwen2.5-0.5B-Instruct (tokenizer) | ~11 MB | F:\unified-sel\topomem\data\models\hf_cache\hub\ |
| all-MiniLM-L6-v2 (embedding) | ~91 MB | 按需下载到 F 盘 |

**总计：~4.85 GB（GGUF 模型）+ 少量缓存**

## 清理 C 盘缓存

如果发现 C 盘有 HuggingFace 缓存，可以安全删除：

```cmd
rmdir /s /q C:\Users\Administrator\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct
rmdir /s /q C:\Users\Administrator\.cache\huggingface\hub\models--sentence-transformers--all-MiniLM-L6-v2
```

## 验证方法

运行测试验证缓存路径：

```bash
F:\unified-sel\.venv\Scripts\python.exe -m pytest F:\unified-sel\topomem\tests\test_infra.py::TestEnvironmentVariables -v
```

所有 4 项环境变量测试应该通过，确认缓存指向 F 盘。
