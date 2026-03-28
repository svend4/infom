"""
KnowledgeMap — полная карта знаний.
Сота из шестиугольных сообществ, связанных фрактальными границами.
Три уровня иерархии:
    L0: GraphNode     — отдельные сущности
    L1: HyperEdge     — кластеры 3-8 нод (Tangram + Heptagram)
    L2: Community     — шестиугольные сообщества (Q6 + Fractal + Octagram)
    L3: KnowledgeMap  — сота всех сообществ
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math

from signatures import (
    voronoi_cells, packing_number, delaunay_graph,
    hamming, median, metric_interval, hamming_ball,
    build_fractal_signature,
)
from graph.node       import GraphNode, GraphEdge
from graph.hyper_edge import HyperEdge, build_hyper_edges
from graph.community  import Community, CommunityBorder


@dataclass
class KnowledgeMap:
    """
    Полная карта знаний — сота из сообществ.
    """
    nodes:       dict[str, GraphNode]     = field(default_factory=dict)
    edges:       list[GraphEdge]          = field(default_factory=list)
    hyper_edges: list[HyperEdge]          = field(default_factory=list)
    communities: dict[str, Community]     = field(default_factory=dict)
    borders:     list[CommunityBorder]    = field(default_factory=list)
    metadata:    dict[str, Any]           = field(default_factory=dict)

    # ── добавление данных ────────────────────────────────────────────────

    def add_node(self, node: GraphNode) -> None:
        self.nodes[node.id] = node

    def add_edge(self, edge: GraphEdge) -> None:
        self.edges.append(edge)

    def add_community(self, community: Community) -> None:
        self.communities[community.id] = community

    # ── построение карты ─────────────────────────────────────────────────

    def build(self, n_communities: int = 8) -> None:
        """
        Полный цикл построения карты знаний:
        1. Проецируем ноды в Q6
        2. Voronoi разбиение → шестиугольные сообщества
        3. Строим гиперрёбра (Tangram)
        4. Строим границы (Fractal)
        5. Граф Делоне между сообществами
        """
        if not self.nodes:
            return

        # 1. Q6 центры через sphere packing
        centers = packing_number(radius=1)[:n_communities]

        # 2. Voronoi — разбиваем ноды по сообществам
        node_list = list(self.nodes.values())
        cells     = voronoi_cells(centers)

        hex_to_nodes: dict[int, list[GraphNode]] = {}
        for node in node_list:
            hid = node.hex_id
            nearest = min(centers, key=lambda c: hamming(hid, c))
            hex_to_nodes.setdefault(nearest, []).append(node)

        # 3. Создаём сообщества
        for center_hex, comm_nodes in hex_to_nodes.items():
            if not comm_nodes:
                continue
            comm_id   = f"comm_{center_hex}"
            comm_edges = [e for e in self.edges
                          if e.source in {n.id for n in comm_nodes}
                          and e.target in {n.id for n in comm_nodes}]

            community = Community(
                id    = comm_id,
                label = f"community_{center_hex}",
                nodes = comm_nodes,
                edges = comm_edges,
            )

            # 2D позиции через PCA-lite (первые 2 компоненты эмбеддинга)
            positions_2d = {
                n.id: (n.embedding[0] if len(n.embedding)>0 else 0.0,
                       n.embedding[1] if len(n.embedding)>1 else 0.0)
                for n in comm_nodes
            }

            community.build_all_signatures(node_2d_positions=positions_2d)
            self.add_community(community)

        # 4. Гиперрёбра
        all_positions = {}
        for node in self.nodes.values():
            all_positions[node.id] = (
                node.embedding[0] if len(node.embedding)>0 else 0.0,
                node.embedding[1] if len(node.embedding)>1 else 0.0,
            )
        self.hyper_edges = build_hyper_edges(
            list(self.nodes.keys()), self.edges, all_positions
        )

        # 5. Границы между сообществами (Делоне)
        comm_list     = list(self.communities.values())
        comm_hex_ids  = [c.hex_id for c in comm_list]
        delaunay_edges = delaunay_graph(comm_hex_ids)

        for (ha, hb) in delaunay_edges:
            ca = next((c for c in comm_list if c.hex_id == ha), None)
            cb = next((c for c in comm_list if c.hex_id == hb), None)
            if ca and cb:
                border = _build_border(ca, cb)
                self.borders.append(border)
                ca.borders.append(border)
                cb.borders.append(border)
                ca.neighbors.append(cb.id)
                cb.neighbors.append(ca.id)

    # ── поиск ────────────────────────────────────────────────────────────

    def find_community(self, node_id: str) -> Community | None:
        for comm in self.communities.values():
            if any(n.id == node_id for n in comm.nodes):
                return comm
        return None

    def communities_in_radius(self, center_comm_id: str,
                               radius: int = 1) -> list[Community]:
        """Все сообщества в Q6-радиусе от данного."""
        center = self.communities.get(center_comm_id)
        if not center:
            return []
        ball_ids = hamming_ball(center.hex_id, radius)
        return [c for c in self.communities.values()
                if c.hex_id in ball_ids and c.id != center_comm_id]

    def path_between(self, comm_a_id: str, comm_b_id: str) -> list[Community]:
        """Q6 кратчайший путь между двумя сообществами."""
        ca = self.communities.get(comm_a_id)
        cb = self.communities.get(comm_b_id)
        if not ca or not cb:
            return []
        interval_ids = metric_interval(ca.hex_id, cb.hex_id)
        hex_to_comm  = {c.hex_id: c for c in self.communities.values()}
        return [hex_to_comm[hid] for hid in interval_ids if hid in hex_to_comm]

    def similar_boundaries(self, border: CommunityBorder,
                            top_k: int = 5) -> list[CommunityBorder]:
        """Найти похожие границы по IFS-сигнатуре."""
        ranked = sorted(
            self.borders,
            key=lambda b: b.similarity(border),
            reverse=True,
        )
        return [b for b in ranked if b is not border][:top_k]

    def communities_by_shape(self, shape_class) -> list[Community]:
        """Все сообщества с заданным Tangram-shape."""
        return [c for c in self.communities.values()
                if c.tangram and c.tangram.shape_class == shape_class]

    def summary(self) -> str:
        lines = [
            f"KnowledgeMap:",
            f"  nodes:       {len(self.nodes)}",
            f"  edges:       {len(self.edges)}",
            f"  hyper_edges: {len(self.hyper_edges)}",
            f"  communities: {len(self.communities)}",
            f"  borders:     {len(self.borders)}",
        ]
        if self.communities:
            shapes: dict[str, int] = {}
            for c in self.communities.values():
                s = c.tangram.shape_class.value if c.tangram else "unknown"
                shapes[s] = shapes.get(s, 0) + 1
            lines.append("  shapes:")
            for s, cnt in sorted(shapes.items()):
                lines.append(f"    {s}: {cnt}")
        return "\n".join(lines)


# ── вспомогательные функции ─────────────────────────────────────────────────

def _build_border(ca: Community, cb: Community) -> CommunityBorder:
    """Построить FractalSignature границы между двумя сообществами."""
    # граничная кривая = интерполяция между центрами через общие ноды
    curve = _boundary_curve(ca, cb)
    return CommunityBorder(
        community_a  = ca.id,
        community_b  = cb.id,
        fractal      = build_fractal_signature(curve),
        shared_nodes = _shared_neighbors(ca, cb),
    )


def _boundary_curve(
    ca: Community,
    cb: Community,
    n_points: int = 16,
) -> list[tuple[float, float]]:
    """
    Приближённая граничная кривая между двумя сообществами.
    Строится как синусоидальное возмущение прямой линии между центрами.
    """
    ea = ca.center_embedding
    eb = cb.center_embedding
    if not ea or not eb:
        return [(float(i), 0.0) for i in range(n_points)]

    x0 = ea[0] if len(ea) > 0 else 0.0
    y0 = ea[1] if len(ea) > 1 else 0.0
    x1 = eb[0] if len(eb) > 0 else 1.0
    y1 = eb[1] if len(eb) > 1 else 0.0

    # добавляем фрактальное возмущение на основе Q6-расстояния
    hd = hamming(ca.hex_id, cb.hex_id)
    amp = hd * 0.05  # амплитуда = Q6 расстояние

    curve = []
    for i in range(n_points):
        t    = i / (n_points - 1)
        x    = x0 + t * (x1 - x0)
        y    = y0 + t * (y1 - y0)
        # перпендикулярное смещение
        perp_x = -(y1 - y0)
        perp_y =  (x1 - x0)
        norm   = math.hypot(perp_x, perp_y) or 1.0
        noise  = amp * math.sin(math.pi * t * hd)
        x     += noise * perp_x / norm
        y     += noise * perp_y / norm
        curve.append((x, y))

    return curve


def _shared_neighbors(ca: Community, cb: Community) -> list[str]:
    a_ids = {n.id for n in ca.nodes}
    b_ids = {n.id for n in cb.nodes}
    return list(a_ids & b_ids)
