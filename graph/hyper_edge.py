"""
HyperEdge — N-арная связь между 3+ нодами.
Геометрически описывается TangramSignature.
Многомерно описывается HeptagramSignature.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from signatures import (
    TangramSignature, build_tangram_signature,
    HeptagramSignature, heptagram_from_edge_weights,
    ShapeClass,
)
from graph.node import GraphEdge


@dataclass
class HyperEdge:
    """
    Гиперребро: связь между N нодами (N >= 3).

    Уровень 1: бинарные рёбра A→B
    Уровень 2: HyperEdge = {A, B, C, ...} + TangramSignature
    """
    id:        str
    nodes:     list[str]           # 3..N нод
    label:     str                 # тема / паттерн / событие
    weight:    float = 1.0
    tangram:   TangramSignature | None = None
    heptagram: HeptagramSignature | None = None
    edges:     list[GraphEdge] = field(default_factory=list)  # составные рёбра
    metadata:  dict[str, Any]  = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def shape(self) -> ShapeClass | None:
        return self.tangram.shape_class if self.tangram else None

    def build_signatures(
        self,
        node_positions: list[tuple[float, float]],
        edge_weights:   list[tuple[str, str, float]] | None = None,
    ) -> None:
        """
        Вычислить геометрические подписи.
        node_positions: 2D-проекции эмбеддингов нод кластера.
        """
        if len(node_positions) >= 3:
            self.tangram = build_tangram_signature(node_positions)

        if edge_weights:
            self.heptagram = heptagram_from_edge_weights(
                self.nodes, edge_weights
            )


def build_hyper_edges(
    nodes:           list[str],
    edges:           list[GraphEdge],
    node_positions:  dict[str, tuple[float, float]],
    min_cluster_size: int = 3,
) -> list[HyperEdge]:
    """
    Автоматически строит гиперрёбра из набора рёбер через поиск клик.
    Простая реализация через пересечение соседей.
    """
    # строим словарь смежности
    adj: dict[str, set[str]] = {n: set() for n in nodes}
    for e in edges:
        adj[e.source].add(e.target)
        if not e.directed:
            adj[e.target].add(e.source)

    # ищем группы взаимосвязанных нод (cliques / dense subgraphs)
    hyper_edges = []
    visited     = set()

    for node in nodes:
        if node in visited:
            continue
        # BFS с ограничением по радиусу
        cluster    = {node}
        frontier   = list(adj[node])
        for nb in frontier:
            shared = adj[node] & adj[nb]
            if len(shared) >= min_cluster_size - 2:
                cluster.add(nb)
            if len(cluster) >= 8:   # max octagram
                break

        if len(cluster) >= min_cluster_size:
            cluster_list = list(cluster)
            positions    = [node_positions.get(n, (0.0, 0.0)) for n in cluster_list]
            edge_w       = [(e.source, e.target, e.weight)
                            for e in edges
                            if e.source in cluster and e.target in cluster]

            he = HyperEdge(
                id    = f"he_{node}",
                nodes = cluster_list,
                label = _label_by_size(len(cluster_list)),
            )
            he.build_signatures(positions, edge_w)
            hyper_edges.append(he)
            visited.update(cluster)

    return hyper_edges


def _label_by_size(n: int) -> str:
    labels = {
        3: "triad",
        4: "quad",
        5: "pentagon_cluster",
        6: "hexagon_community",
        7: "heptagram_group",
        8: "octagram_domain",
    }
    return labels.get(n, f"poly_{n}_cluster")
