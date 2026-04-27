"""
weight_graph/extractor.py — 从 HuggingFace 模型提取权重矩阵

职责：
  1. 加载预训练模型（或只加载 state_dict）
  2. 按层类型和索引筛选权重张量
  3. 输出标准化的 WeightMatrix 列表

实现者注意：
  - 不同模型架构的参数命名不同，需要自适应解析
  - Qwen2 的 MLP 是 gate_proj + up_proj + down_proj（SwiGLU）
  - LLaMA 系列类似
  - 需要处理 GQA（Grouped Query Attention）的 K/V 共享
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig

from weight_graph.config import ExtractionConfig


@dataclass
class WeightMatrix:
    """一个权重矩阵及其元数据。"""

    name: str                    # 参数全名，如 "model.layers.0.mlp.gate_proj.weight"
    layer_index: int             # 所属 Transformer 层索引
    component: str               # mlp_gate / mlp_up / mlp_down / attn_q / attn_k / attn_v / attn_o
    weight: np.ndarray           # shape [d_out, d_in]，从 PyTorch 转 numpy
    d_in: int
    d_out: int


# ─────────────────────────────────────────────────────────
# Component name mapping
# ─────────────────────────────────────────────────────────

_COMPONENT_MAP = {
    "gate_proj": "mlp_gate",
    "up_proj": "mlp_up",
    "down_proj": "mlp_down",
    "q_proj": "attn_q",
    "k_proj": "attn_k",
    "v_proj": "attn_v",
    "o_proj": "attn_o",
}

# Patterns that indicate a layer parameter
# Each returns (layer_idx, component)
_LAYER_PATTERNS = [
    # Qwen2 / LLaMA style: model.layers.N.mlp.gate_proj
    re.compile(r"model\.layers\.(\d+)\.(mlp)\.(gate_proj|up_proj|down_proj)"),
    # Qwen2 / LLaMA style: model.layers.N.attn.q_proj (or self_attn.q_proj)
    re.compile(r"model\.layers\.(\d+)\.(?:self_attn|attention)\.(q_proj|k_proj|v_proj|o_proj)"),
    re.compile(r"model\.layers\.(\d+)\.(q_proj|k_proj|v_proj|o_proj)"),
    # Older dict name style: layers.N.mlp.gate_proj
    re.compile(r"layers\.(\d+)\.(mlp|self_attn|attention)\.(gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)"),
    # LLaMA style: h.N.gate_proj
    re.compile(r"\.h\.(\d+)\.(gate_proj|up_proj|down_proj|q_proj|k_proj|v_proj|o_proj)"),
]


class WeightExtractor:
    """
    从 HuggingFace 预训练模型提取权重矩阵。

    用法:
        extractor = WeightExtractor(config)
        matrices = extractor.extract()
        # matrices: List[WeightMatrix]
    """

    def __init__(self, config: ExtractionConfig):
        self.config = config
        self._state_dict = None
        self._model_info = None

    def _load_state_dict(self) -> Dict[str, torch.Tensor]:
        """懒加载预训练模型 state_dict。"""
        if self._state_dict is None:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(
                self.config.model_name,
                dtype=torch.float32,
                trust_remote_code=True,
            )
            self._state_dict = {k: v.float() for k, v in model.state_dict().items()}
            del model
        return self._state_dict

    def _load_random_state_dict(self, seed: int | None = None) -> Dict[str, torch.Tensor]:
        """
        加载随机初始化模型的 state_dict（相同架构）。

        参数:
            seed: 随机种子（None = 不设置种子）
        """
        if seed is not None:
            import random
            import numpy as np
            torch.manual_seed(seed)
            random.seed(seed)
            np.random.seed(seed)

        from transformers import AutoConfig, AutoModel
        config = AutoConfig.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
        )
        model = AutoModel.from_config(
            config,
            dtype=torch.float32,
        )
        state_dict = {k: v.float() for k, v in model.state_dict().items()}
        del model
        return state_dict

    def extract(self) -> List[WeightMatrix]:
        """提取所有符合条件的权重矩阵。"""
        state_dict = self._load_state_dict()
        matrices = []

        for name, tensor in state_dict.items():
            parsed = self._parse_param_name(name)
            if parsed is None:
                continue

            layer_idx, component = parsed

            # 按 layer_types 过滤
            if self.config.layer_types:
                type_map = {"mlp": "mlp", "attention": "attn", "all": "all"}
                comp_type = type_map.get(component.split("_")[0], "")
                if comp_type not in self.config.layer_types and component not in self.config.layer_types:
                    continue

            # 按 layer_indices 过滤
            if self.config.layer_indices is not None:
                if layer_idx not in self.config.layer_indices:
                    continue

            weight = tensor.cpu().numpy()
            d_out, d_in = weight.shape

            matrices.append(WeightMatrix(
                name=name,
                layer_index=layer_idx,
                component=component,
                weight=weight,
                d_in=d_in,
                d_out=d_out,
            ))

        return matrices

    def extract_single_layer(self, layer_index: int) -> List[WeightMatrix]:
        """只提取指定层的权重矩阵。Phase 1 用这个。"""
        state_dict = self._load_state_dict()
        matrices = []

        # 只提取 MLP 相关的组件
        mlp_components = {"mlp_gate", "mlp_up", "mlp_down"}

        for name, tensor in state_dict.items():
            parsed = self._parse_param_name(name)
            if parsed is None:
                continue

            layer_idx, component = parsed
            if layer_idx != layer_index:
                continue
            if component not in mlp_components:
                continue

            weight = tensor.cpu().numpy()
            # 跳过非 2D 张量（如 embedding）
            if weight.ndim != 2:
                continue
            d_out, d_in = weight.shape

            matrices.append(WeightMatrix(
                name=name,
                layer_index=layer_idx,
                component=component,
                weight=weight,
                d_in=d_in,
                d_out=d_out,
            ))

        return matrices

    def extract_random_init(self, seed: int | None = None) -> List[WeightMatrix]:
        """
        提取随机初始化模型的权重矩阵（相同架构，用于对比实验）。

        参数:
            seed: 随机种子（None = 使用 config 中的 seed 或系统默认）

        返回: List[WeightMatrix]，格式同 extract()
        """
        use_seed = seed if seed is not None else self.config.seed
        state_dict = self._load_random_state_dict(use_seed)
        matrices = []

        for name, tensor in state_dict.items():
            parsed = self._parse_param_name(name)
            if parsed is None:
                continue

            layer_idx, component = parsed

            # 按 layer_types 过滤
            if self.config.layer_types:
                type_map = {"mlp": "mlp", "attention": "attn", "all": "all"}
                comp_type = type_map.get(component.split("_")[0], "")
                if comp_type not in self.config.layer_types and component not in self.config.layer_types:
                    continue

            # 按 layer_indices 过滤
            if self.config.layer_indices is not None:
                if layer_idx not in self.config.layer_indices:
                    continue

            weight = tensor.cpu().numpy()
            # 跳过非 2D 张量（如 embedding）
            if weight.ndim != 2:
                continue
            d_out, d_in = weight.shape

            matrices.append(WeightMatrix(
                name=name,
                layer_index=layer_idx,
                component=component,
                weight=weight,
                d_in=d_in,
                d_out=d_out,
            ))

        return matrices

    def get_model_info(self) -> Dict:
        """返回模型基本信息（不加载完整权重，只用 AutoConfig）。"""
        if self._model_info is None:
            config = AutoConfig.from_pretrained(
                self.config.model_name,
                trust_remote_code=True,
            )
            self._model_info = {
                "model_name": self.config.model_name,
                "num_layers": getattr(config, "num_hidden_layers", 0),
                "hidden_size": getattr(config, "hidden_size", 0),
                "intermediate_size": getattr(config, "intermediate_size", 0),
                "num_attention_heads": getattr(config, "num_attention_heads", 0),
                "num_kv_heads": getattr(config, "num_key_value_heads", 0),
                "vocab_size": getattr(config, "vocab_size", 0),
            }
        return self._model_info

    @staticmethod
    def _parse_param_name(name: str) -> Optional[Tuple[int, str]]:
        """
        解析参数名，返回 (layer_index, component_type)。

        示例:
            "model.layers.5.mlp.gate_proj.weight"  → (5, "mlp_gate")
            "model.layers.3.self_attn.q_proj.weight" → (3, "attn_q")
            "model.embed_tokens.weight"             → None（非层参数）
        """
        for pattern in _LAYER_PATTERNS:
            m = pattern.search(name)
            if m:
                layer_idx = int(m.group(1))
                num_groups = len(m.groups())
                if num_groups == 3:
                    # Patterns with layer-type prefix: (layer, type, component)
                    component_raw = m.group(3)
                else:
                    # Patterns without prefix: (layer, component)
                    component_raw = m.group(2)
                component = _COMPONENT_MAP.get(component_raw, component_raw)
                return (layer_idx, component)
        return None
