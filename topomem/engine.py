"""
topomem/engine.py — 推理引擎集成

轻量 LLM 推理引擎，支持记忆上下文注入的生成式推理。

主选后端：llama-cpp-python（GGUF 量化模型）
备选后端：transformers（HuggingFace 模型）

设计来源：
- SPEC_ENGINE.md: 推理引擎集成完整规格
- sentence-transformers: EmbeddingManager（已在 embedding.py 实现）
- transformers: Qwen2.5-0.5B-Instruct fallback 后端
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

from topomem.config import EngineConfig


logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Prompt 模板
# ------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """You are a precise and consistent reasoning assistant.
You answer questions based on the provided context.
If the context doesn't contain relevant information, say so clearly.
Be concise and factual."""


def format_memory_context(memories: List[dict]) -> str:
    """将检索到的记忆格式化为 LLM 可理解的上下文。

    参数：
        memories: List[dict]，每个包含：
            - content: str
            - cluster_id: int
            - access_count: int

    返回：
        格式化的上下文字符串
    """
    if not memories:
        return ""

    lines = ["--- Relevant Knowledge ---"]
    for i, mem in enumerate(memories, 1):
        content = mem.get("content", "")
        cluster_id = mem.get("cluster_id", -1)
        access_count = mem.get("access_count", 0)

        cluster_info = f"Cluster {cluster_id}" if cluster_id >= 0 else "Unclassified"
        lines.append(
            f"[{i}] {content}\n"
            f"(Relevance: {cluster_info}, Accessed: {access_count} times)"
        )
    lines.append("--- End of Knowledge ---")
    return "\n\n".join(lines)


def build_prompt(
    user_query: str,
    context: Optional[List[dict]] = None,
    system_prompt: Optional[str] = None,
) -> str:
    """组装完整 prompt。

    格式：
    [System]
    {system_prompt}

    [User]
    {memory_context}

    Based on the above knowledge, answer the following question:
    {user_query}
    """
    sys_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT

    parts = [f"[System]\n{sys_prompt}", "[User]"]

    if context:
        ctx = format_memory_context(context)
        parts.append(ctx)
        parts.append(f"Based on the above knowledge, answer the following question:\n{user_query}")
    else:
        parts.append(user_query)

    return "\n\n".join(parts)


# ------------------------------------------------------------------
# ReasoningEngine 实现
# ------------------------------------------------------------------

class ReasoningEngine:
    """轻量 LLM 推理引擎。

    主选后端：llama-cpp-python（GGUF）
    备选后端：transformers（HuggingFace）
    """

    def __init__(
        self,
        config: Optional[EngineConfig] = None,
    ):
        """
        初始化推理引擎。

        后端选择逻辑：
        1. 如果 config.use_fallback=True → 使用 transformers
        2. 否则尝试 llama-cpp-python
        3. 如果 llama-cpp 不可用 → fallback 到 transformers
        """
        self.config = config or EngineConfig()
        self._backend = None  # "llama_cpp" or "transformers"
        self._model = None
        self._tokenizer = None

        self._initialize_backend()

    def _initialize_backend(self) -> None:
        """初始化后端。"""
        # 强制 fallback
        if self.config.use_fallback:
            self._init_transformers()
            return

        # 尝试 llama-cpp-python
        try:
            import llama_cpp
            model_path = Path(self.config.model_path)
            if model_path.exists():
                self._init_llama_cpp()
                return
            else:
                logger.warning(
                    f"GGUF model not found at {model_path}, "
                    f"falling back to transformers."
                )
        except ImportError:
            logger.info(
                "llama-cpp-python not available, "
                "falling back to transformers."
            )

        # Fallback 到 transformers
        self._init_transformers()

    def _init_llama_cpp(self) -> None:
        """初始化 llama-cpp-python 后端。"""
        from llama_cpp import Llama

        self._model = Llama(
            model_path=self.config.model_path,
            n_ctx=self.config.n_ctx,
            n_threads=self.config.n_threads,
            verbose=False,
        )
        self._backend = "llama_cpp"
        logger.info(f"Loaded GGUF model: {self.config.model_path}")

    def _init_transformers(self) -> None:
        """初始化 transformers 后端。"""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model_name = self.config.fallback_model_name
        logger.info(f"Loading transformers model: {model_name}")

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 确保 tokenizer 有 pad token
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # 不使用 device_map，直接指定 device
        device = torch.device("cpu")
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        self._model = self._model.to(device)
        self._model.eval()

        self._backend = "transformers"
        logger.info(f"Loaded transformers model: {model_name}")

    # ------------------------------------------------------------------
    # 核心 API: generate
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        context: Optional[List[dict]] = None,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """核心推理方法。

        参数：
            prompt: 用户查询
            context: 检索到的记忆列表（可选）
            system_prompt: 自定义系统提示（可选）
            max_tokens: 最大生成 token 数
            temperature: 温度参数

        返回：
            生成的文本
        """
        if not prompt:
            raise ValueError("prompt cannot be empty")

        full_prompt = build_prompt(prompt, context, system_prompt)
        max_tok = max_tokens or self.config.max_tokens
        temp = temperature if temperature is not None else self.config.temperature

        if self._backend == "llama_cpp":
            return self._generate_llama_cpp(full_prompt, max_tok, temp)
        else:
            return self._generate_transformers(full_prompt, max_tok, temp)

    def _generate_llama_cpp(
        self,
        full_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """使用 llama-cpp-python 生成。"""
        result = self._model.create_chat_completion(
            messages=[
                {"role": "user", "content": full_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return result["choices"][0]["message"]["content"].strip()

    def _generate_transformers(
        self,
        full_prompt: str,
        max_tokens: int,
        temperature: float,
    ) -> str:
        """使用 transformers 生成。"""
        import torch

        inputs = self._tokenizer(
            full_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.n_ctx - max_tokens,
        )

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0.0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        # 解码（跳过输入部分）
        input_len = inputs["input_ids"].shape[1]
        generated_ids = outputs[0][input_len:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def estimate_tokens(self, text: str) -> int:
        """粗略估算文本的 token 数。

        如果有 tokenizer 则使用精确计数，
        否则使用启发式估计。
        """
        if self._tokenizer is not None:
            return len(self._tokenizer.encode(text))

        # 启发式：英文 ~4 字符/token，中文 ~2 字符/token
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        other_chars = len(text) - chinese_chars
        return chinese_chars // 2 + other_chars // 4

    def truncate_context(
        self,
        memories: List[dict],
        max_context_tokens: int = 1024,
    ) -> List[dict]:
        """如果记忆太多，截断到 token 预算内。

        策略：
        1. 从最相关的开始累积 token 数
        2. 超过预算则截断
        3. 保证至少保留 1 条记忆

        参数：
            memories: 记忆列表（按相关性排序）
            max_context_tokens: 最大 token 预算

        返回：
            截断后的记忆列表
        """
        if not memories:
            return []

        # 保证至少保留 1 条
        if len(memories) == 1:
            return memories

        accumulated = []
        token_count = 0

        for mem in memories:
            ctx = format_memory_context([mem])
            mem_tokens = self.estimate_tokens(ctx)

            if token_count + mem_tokens > max_context_tokens:
                break

            accumulated.append(mem)
            token_count += mem_tokens

        return accumulated if accumulated else memories[:1]

    def unload(self) -> None:
        """释放模型，回收内存。"""
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        import gc
        gc.collect()

    @property
    def backend(self) -> str:
        """返回当前后端名称。"""
        return self._backend or "unknown"

    def __repr__(self) -> str:
        return (
            f"ReasoningEngine(backend='{self.backend}', "
            f"max_tokens={self.config.max_tokens}, "
            f"temperature={self.config.temperature})"
        )


# ------------------------------------------------------------------
# 知识提取（便捷函数）
# ------------------------------------------------------------------

def extract_knowledge(
    user_query: str,
    response: str,
) -> Optional[str]:
    """从模型回答中提取值得存储的新知识。

    MVP 实现（简单规则）：
    1. 如果 response 长度 < 20 字符 → None
    2. 如果 response 包含 "I don't know" / "not sure" → None
    3. 否则返回合并的 Q+A

    参数：
        user_query: 用户查询
        response: 模型回答

    返回：
        新知识字符串，或 None
    """
    if not response or len(response.strip()) < 20:
        return None

    response_lower = response.lower()
    if "i don't know" in response_lower or "not sure" in response_lower:
        return None

    return f"Q: {user_query}\nA: {response}"
