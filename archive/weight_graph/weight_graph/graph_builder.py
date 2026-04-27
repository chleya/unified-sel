"""
weight_graph/graph_builder.py — 权重矩阵 → 有向图（核心模块）

职责：
  1. 单层权重矩阵 → 二部有向图
  2. 多层堆叠 → 全模型有向图（含残差连接）
  3. Attention 层折叠为超节点

节点命名约定：
  MLP:  "L{layer}_mlp_{component}_{neuron_idx}"
        例: "L0_mlp_gate_42" = 第0层 gate_proj 的第42个输出神经元

  Attn: "L{layer}_attn_head_{head_idx}" （超节点模式）
        或展开为 "L{layer}_attn_{qkvo}_{idx}"

  残差: "R_{dim_idx}" = residual stream 的第 dim_idx 维
        （全局共享，跨层连接的枢纽）

边属性：
  - weight: float，权重绝对值
  - layer: int，所属层
  - component: str，来源组件
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from weight_graph.config import GraphBuildConfig
from weight_graph.extractor import WeightMatrix


class WeightGraph:
    """
    权重图的内部表示。

    不直接用 networkx.DiGraph 作为内部存储（太慢），
    而是用邻接表 + numpy 数组，需要 networkx 时再转换。

    属性：
        nodes: Dict[str, Dict]         — 节点名 → 属性
        edges: List[Tuple[str,str,Dict]] — (src, dst, {weight, layer, ...})
        node_to_idx: Dict[str, int]    — 节点名 → 整数索引（用于矩阵运算）
    """

    def __init__(self):
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[Tuple[str, str, Dict]] = []
        self.node_to_idx: Dict[str, int] = {}
        self._next_idx = 0
        self.io_in_nodes: List[str] = []
        self.io_out_nodes: List[str] = []

    def add_node(self, name: str, **attrs) -> None:
        """添加节点（如已存在则更新属性）。"""
        if name not in self.nodes:
            self.node_to_idx[name] = self._next_idx
            self._next_idx += 1
        self.nodes[name] = attrs

    def add_edge(self, src: str, dst: str, **attrs) -> None:
        """添加有向边。"""
        self.edges.append((src, dst, attrs))

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def num_edges(self) -> int:
        return len(self.edges)

    def to_networkx(self):
        """转换为 networkx.DiGraph。"""
        import networkx as nx

        G = nx.DiGraph()
        for name, attrs in self.nodes.items():
            G.add_node(name, **attrs)
        for src, dst, attrs in self.edges:
            G.add_edge(src, dst, **attrs)
        return G

    def to_sparse_adjacency(self) -> Tuple:
        """转换为 scipy 稀疏邻接矩阵。用于大规模图的快速矩阵运算。"""
        from scipy.sparse import csr_matrix

        n = len(self.nodes)
        if n == 0:
            return csr_matrix((0, 0)), {}

        idx_map = {name: idx for name, idx in self.node_to_idx.items()}
        rows, cols, data = [], [], []

        for src, dst, attrs in self.edges:
            rows.append(idx_map[src])
            cols.append(idx_map[dst])
            data.append(float(attrs.get("weight", 1.0)))

        mat = csr_matrix((data, (rows, cols)), shape=(n, n), dtype=float)
        idx_to_name = {idx: name for name, idx in idx_map.items()}
        return mat, idx_to_name

    def summary(self) -> Dict:
        """返回图的基础统计。"""
        return {
            "num_nodes": self.num_nodes,
            "num_edges": self.num_edges,
            "density": self.num_edges / max(1, self.num_nodes * (self.num_nodes - 1)),
        }


def _merge_into(dst: WeightGraph, src: WeightGraph) -> None:
    """
    将 src 图的所有节点和边合并到 dst 图（in-place）。
    src 自身的 io_in_nodes / io_out_nodes 不影响 dst。
    """
    for name, attrs in src.nodes.items():
        if name not in dst.nodes:
            dst.node_to_idx[name] = dst._next_idx
            dst._next_idx += 1
        dst.nodes[name] = attrs
    for edge in src.edges:
        dst.edges.append(edge)


class GraphBuilder:
    """
    从 WeightMatrix 列表构建有向图。

    两种模式：
    1. build_single_layer() — 单层二部图（Phase 1）
    2. build_full_model()   — 全模型图含残差（Phase 3）
    """

    def __init__(self, config: GraphBuildConfig):
        self.config = config

    def build_single_layer(self, matrix: WeightMatrix) -> WeightGraph:
        """单层权重矩阵 → 二部有向图。"""
        from weight_graph.utils import sparsify_percentile, sparsify_topk, sparsify_sigma

        graph = WeightGraph()
        layer = matrix.layer_index
        component = matrix.component
        d_in = matrix.d_in
        d_out = matrix.d_out

        # 节点：输入侧（in_{idx}）和输出侧（out_{idx}）
        for i in range(d_in):
            graph.add_node(f"L{layer}_{component}_in_{i}", layer=layer, component=component, side="in")
        for j in range(d_out):
            graph.add_node(f"L{layer}_{component}_out_{j}", layer=layer, component=component, side="out")

        # 稀疏化
        W_sparse = self._sparsify(matrix.weight)
        W_abs = np.abs(W_sparse)

        # 边：input_i → output_j，边权 = |W[j,i]|
        for i in range(d_in):
            for j in range(d_out):
                w = W_abs[j, i]
                if w > 0:
                    graph.add_edge(
                        f"L{layer}_{component}_in_{i}",
                        f"L{layer}_{component}_out_{j}",
                        weight=float(w),
                        layer=layer,
                        component=component,
                    )

        return graph

    def build_full_model(self, matrices: List[WeightMatrix]) -> WeightGraph:
        """
        全模型权重矩阵列表 → 完整有向图。

        步骤：
        1. 按层组织矩阵（MLP: gate/up/down, Attn: q/k/v/o）
        2. 对每层 MLP 调用 _build_mlp_subgraph()
        3. 对每层 Attention 调用 _build_attention_subgraph()
        4. 添加层间连接（每层输出 → 下一层 gate/up 的对应输入维度）
        5. 如果 config.add_residual，添加残差连接
        6. 合并所有子图
        """
        # Organize matrices by layer and component type
        by_layer: Dict[int, Dict[str, WeightMatrix]] = {}
        for m in matrices:
            by_layer.setdefault(m.layer_index, {})[m.component] = m

        num_layers = max(by_layer.keys()) + 1 if by_layer else 0
        hidden_size = None
        for layer_idx, components in by_layer.items():
            if "mlp_gate" in components:
                hidden_size = components["mlp_gate"].d_in
                break
            if "attn_q" in components:
                hidden_size = components["attn_q"].d_in
                break

        # Build combined graph
        graph = WeightGraph()

        # Track per-layer node names for cross-layer connections
        layer_io_nodes: Dict[int, Tuple[List[str], List[str]]] = {}
        # (input_node_names, output_node_names) per layer

        # Process each layer
        for layer_idx in sorted(by_layer.keys()):
            components = by_layer[layer_idx]

            if "mlp_gate" in components and "mlp_up" in components and "mlp_down" in components:
                mlp_g = self._build_mlp_subgraph(
                    components["mlp_gate"],
                    components["mlp_up"],
                    components["mlp_down"],
                )
                _merge_into(graph, mlp_g)
                layer_io_nodes[layer_idx] = (
                    mlp_g.io_in_nodes,
                    mlp_g.io_out_nodes,
                )
            elif "attn_q" in components:
                q = components.get("attn_q")
                k = components.get("attn_k")
                v = components.get("attn_v")
                o = components.get("attn_o")
                attn_g = self._build_attention_subgraph(q, k, v, o)
                _merge_into(graph, attn_g)
                layer_io_nodes[layer_idx] = (
                    attn_g.io_in_nodes,
                    attn_g.io_out_nodes,
                )

        # Cross-layer connections: output of layer L → input of layer L+1
        sorted_layers = sorted(layer_io_nodes.keys())
        for i in range(len(sorted_layers) - 1):
            l_curr = sorted_layers[i]
            l_next = sorted_layers[i + 1]
            out_nodes_curr, in_nodes_next = (
                layer_io_nodes[l_curr][1],
                layer_io_nodes[l_next][0],
            )
            # Connect each output to corresponding input (same dimension index)
            for out_n, in_n in zip(out_nodes_curr, in_nodes_next):
                if out_n and in_n:
                    graph.add_edge(
                        out_n, in_n,
                        weight=1.0,
                        layer=l_curr,
                        component="cross_layer",
                    )

        # Residual connections
        if self.config.add_residual and hidden_size is not None:
            self._add_residual_connections(graph, hidden_size, num_layers)

        return graph

    def _build_mlp_subgraph(
        self,
        gate: WeightMatrix,
        up: WeightMatrix,
        down: WeightMatrix,
    ) -> WeightGraph:
        """
        构建单层 MLP 的子图（SwiGLU 架构）。

        节点类型：
        - L{layer}_mlp_in_{i}  : input neurons (hidden dim)
        - L{layer}_mlp_gate_{j}: gate projection outputs (intermediate dim)
        - L{layer}_mlp_up_{j}  : up projection outputs (intermediate dim)
        - L{layer}_mlp_mid_{j} : element-wise product of gate[j] * up[j]
        - L{layer}_mlp_out_{k} : output neurons (hidden dim)

        信息流：
        in_i → gate_j (weight: |W_gate[j,i]|)
        in_i → up_j   (weight: |W_up[j,i]|)
        gate_j → mid_j (weight: 1.0, element-wise)
        up_j → mid_j   (weight: 1.0, element-wise)
        mid_j → out_k  (weight: |W_down[k,j]|)
        """
        graph = WeightGraph()
        layer = gate.layer_index

        d_in = gate.d_in
        d_intermediate = gate.d_out
        d_out = down.d_out

        # Create nodes
        in_nodes = [f"L{layer}_mlp_in_{i}" for i in range(d_in)]
        gate_nodes = [f"L{layer}_mlp_gate_{j}" for j in range(d_intermediate)]
        up_nodes = [f"L{layer}_mlp_up_{j}" for j in range(d_intermediate)]
        mid_nodes = [f"L{layer}_mlp_mid_{j}" for j in range(d_intermediate)]
        out_nodes = [f"L{layer}_mlp_out_{k}" for k in range(d_out)]

        for n in in_nodes:
            graph.add_node(n, layer=layer, component="mlp", side="in")
        for n in gate_nodes:
            graph.add_node(n, layer=layer, component="mlp_gate", side="gate")
        for n in up_nodes:
            graph.add_node(n, layer=layer, component="mlp_up", side="up")
        for n in mid_nodes:
            graph.add_node(n, layer=layer, component="mlp_mid", side="mid")
        for n in out_nodes:
            graph.add_node(n, layer=layer, component="mlp", side="out")

        # Sparse gate weights
        W_gate = self._sparsify(gate.weight)
        W_up = self._sparsify(up.weight)
        W_down = self._sparsify(down.weight)

        # Edges: in → gate, in → up
        for j in range(d_intermediate):
            for i in range(d_in):
                w_g = abs(W_gate[j, i])
                if w_g > 0:
                    graph.add_edge(in_nodes[i], gate_nodes[j], weight=float(w_g), layer=layer, component="mlp_gate")
                w_u = abs(W_up[j, i])
                if w_u > 0:
                    graph.add_edge(in_nodes[i], up_nodes[j], weight=float(w_u), layer=layer, component="mlp_up")

        # Edges: gate → mid, up → mid (element-wise product)
        for j in range(d_intermediate):
            graph.add_edge(gate_nodes[j], mid_nodes[j], weight=1.0, layer=layer, component="mlp_mid")
            graph.add_edge(up_nodes[j], mid_nodes[j], weight=1.0, layer=layer, component="mlp_mid")

        # Edges: mid → out
        for k in range(d_out):
            for j in range(d_intermediate):
                w_d = abs(W_down[k, j])
                if w_d > 0:
                    graph.add_edge(mid_nodes[j], out_nodes[k], weight=float(w_d), layer=layer, component="mlp_down")

        graph.io_in_nodes = in_nodes
        graph.io_out_nodes = out_nodes
        return graph

    def _build_attention_subgraph(
        self,
        q: WeightMatrix,
        k: WeightMatrix,
        v: WeightMatrix,
        o: WeightMatrix,
    ) -> WeightGraph:
        """
        构建单层 Attention 的子图。

        如果 config.collapse_attention:
            每个 head 折叠为一个超节点 "L{layer}_attn_head_{h}"
            边权 = 该 head 的 Q/K/V/O 权重范数的几何平均

        GQA 处理：Qwen2 的 num_kv_heads < num_q_heads
        K/V heads 被 group 起来，多个 Q heads 共享同一个 K/V head。
        折叠时：每个 Q head 单独处理，其 K/V 权重取对应 group 的均值。
        """
        graph = WeightGraph()
        layer = q.layer_index

        if q is None:
            return graph

        hidden_size = q.d_in
        num_q_heads = q.d_out
        num_kv_heads = k.d_out if k is not None else num_q_heads
        num_o_heads = o.d_out if o is not None else num_q_heads

        head_dim = hidden_size // num_q_heads

        if self.config.collapse_attention:
            # Collapsed supernode representation
            # One node per Q head (h), edges weighted by combined Q/K/V/O strength
            out_nodes = [f"L{layer}_attn_head_{h}" for h in range(num_q_heads)]

            for n in out_nodes:
                graph.add_node(n, layer=layer, component="attn", side="out")

            W_q_sparse = self._sparsify(q.weight)
            W_k_sparse = self._sparsify(k.weight) if k is not None else None
            W_v_sparse = self._sparsify(v.weight) if v is not None else None
            W_o_sparse = self._sparsify(o.weight) if o is not None else None

            # Geometric mean of ||W_q||_F * ||W_k||_F * ||W_v||_F * ||W_o||_F
            q_norms = []
            for h in range(num_q_heads):
                start, end = h * head_dim, (h + 1) * head_dim
                q_head_norm = float(np.linalg.norm(W_q_sparse[start:end, :], "fro"))

                # K/V head for this Q head (GQA: multiple Q heads share same K/V head)
                if num_kv_heads < num_q_heads:
                    kv_idx = h * num_kv_heads // num_q_heads
                else:
                    kv_idx = h
                kv_start, kv_end = kv_idx * head_dim, (kv_idx + 1) * head_dim
                k_norm = float(np.linalg.norm(W_k_sparse[kv_start:kv_end, :], "fro")) if W_k_sparse is not None else 1.0
                v_norm = float(np.linalg.norm(W_v_sparse[kv_start:kv_end, :], "fro")) if W_v_sparse is not None else 1.0

                o_head_norm = float(np.linalg.norm(W_o_sparse[:, start:end], "fro")) if W_o_sparse is not None else 1.0

                combined_weight = (q_norm * k_norm * v_norm * o_head_norm) ** 0.25
                out_nodes_h = out_nodes[h]

                # Input → attention head (all input dimensions)
                for i in range(hidden_size):
                    in_n = f"L{layer}_attn_in_{i}"
                    if in_n not in graph.nodes:
                        graph.add_node(in_n, layer=layer, component="attn", side="in")
                    graph.add_edge(
                        in_n, out_nodes_h,
                        weight=combined_weight / hidden_size,
                        layer=layer,
                        component="attn",
                    )

                # Attention head → output
                out_node = f"L{layer}_attn_out_{h}"
                graph.add_node(out_node, layer=layer, component="attn", side="out")
                graph.add_edge(
                    out_nodes_h, out_node,
                    weight=1.0,
                    layer=layer,
                    component="attn_o",
                )

            graph.io_in_nodes = [f"L{layer}_attn_in_{i}" for i in range(hidden_size)]
            graph.io_out_nodes = [f"L{layer}_attn_out_{h}" for h in range(num_q_heads)]
        else:
            # Expanded representation (q/k/v/o as separate bipartite graphs)
            for component, matrix in [("q", q), ("k", k), ("v", v), ("o", o)]:
                if matrix is None:
                    continue
                self._build_attention_projection(graph, matrix, layer, component)

            graph.io_in_nodes = [f"L{layer}_attn_q_in_{i}" for i in range(hidden_size)]
            graph.io_out_nodes = [f"L{layer}_attn_o_out_{h}" for h in range(num_o_heads)]

        return graph

    def _build_attention_projection(
        self,
        graph: WeightGraph,
        proj: WeightMatrix,
        layer: int,
        component: str,
    ) -> None:
        """为单个 Q/K/V/O 投影构建二部子图（expanded attention 模式）。"""
        d_out, d_in = proj.weight.shape
        W_sparse = self._sparsify(proj.weight)

        in_prefix = f"L{layer}_attn_{component}"
        for i in range(d_in):
            graph.add_node(f"{in_prefix}_in_{i}", layer=layer, component=f"attn_{component}", side="in")
        for j in range(d_out):
            graph.add_node(f"{in_prefix}_out_{j}", layer=layer, component=f"attn_{component}", side="out")

        for j in range(d_out):
            for i in range(d_in):
                w = abs(W_sparse[j, i])
                if w > 0:
                    graph.add_edge(
                        f"{in_prefix}_in_{i}",
                        f"{in_prefix}_out_{j}",
                        weight=float(w),
                        layer=layer,
                        component=f"attn_{component}",
                    )

    def _add_residual_connections(self, graph: WeightGraph, hidden_size: int, num_layers: int) -> None:
        """
        添加残差连接。

        对每层 l 和每个维度 d ∈ [0, hidden_size)：
        - 创建层特定的残差节点 "L{l}_res_{d}"
        - 层输入从残差节点读：L{l}_res_{d} → L{l}_mlp_in_{d}（或 attn_in_{d}）
        - 层输出写回残差节点：L{l}_mlp_out_{d} → L{l}_res_{d}

        相邻层之间通过残差形成隐式路径：
        L{l}_mlp_out_d → L{l}_res_d → L{l+1}_mlp_in_d → ...
        → 通过 cross_layer 连接形成环路

        注意：残差节点是 per-layer（而非全局），避免单一巨型 SCC。
        """
        for layer in range(num_layers):
            res_nodes = [f"L{layer}_res_{d}" for d in range(hidden_size)]
            for n in res_nodes:
                graph.add_node(n, layer=layer, component="residual", side="residual")

            # Find this layer's in/out nodes
            in_nodes = [f"L{layer}_mlp_in_{d}" for d in range(hidden_size) if f"L{layer}_mlp_in_{d}" in graph.nodes]
            out_nodes = [f"L{layer}_mlp_out_{d}" for d in range(hidden_size) if f"L{layer}_mlp_out_{d}" in graph.nodes]

            # Also check attention nodes
            if not in_nodes:
                in_nodes = [f"L{layer}_attn_in_{d}" for d in range(hidden_size) if f"L{layer}_attn_in_{d}" in graph.nodes]
            if not out_nodes:
                out_nodes = [f"L{layer}_attn_out_{d}" for d in range(hidden_size) if f"L{layer}_attn_out_{d}" in graph.nodes]

            w = float(self.config.residual_weight)
            for d, rn in enumerate(res_nodes):
                if d < len(in_nodes) and in_nodes[d] in graph.nodes:
                    graph.add_edge(rn, in_nodes[d], weight=w, layer=layer, component="residual")
                if d < len(out_nodes) and out_nodes[d] in graph.nodes:
                    graph.add_edge(out_nodes[d], rn, weight=w, layer=layer, component="residual")

    def _sparsify(self, W: np.ndarray) -> np.ndarray:
        """根据 config 选择稀疏化方法。"""
        from weight_graph.utils import sparsify_percentile, sparsify_topk, sparsify_sigma

        method = self.config.sparsify_method
        if method == "percentile":
            return sparsify_percentile(W, self.config.percentile)
        elif method == "topk":
            return sparsify_topk(W, self.config.topk)
        elif method == "sigma":
            return sparsify_sigma(W, self.config.n_sigma)
        else:
            raise ValueError(f"Unknown sparsify_method: {method}")

    def _sparsify(self, W: np.ndarray) -> np.ndarray:
        """根据 config 选择稀疏化方法。"""
        from weight_graph.utils import sparsify_percentile, sparsify_topk, sparsify_sigma

        method = self.config.sparsify_method
        if method == "percentile":
            return sparsify_percentile(W, self.config.percentile)
        elif method == "topk":
            return sparsify_topk(W, self.config.topk)
        elif method == "sigma":
            return sparsify_sigma(W, self.config.n_sigma)
        else:
            raise ValueError(f"Unknown sparsify_method: {method}")
