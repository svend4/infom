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


# ── PCA-lite (power iteration, stdlib only) ──────────────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _sub_mean(vecs: list[list[float]]) -> list[list[float]]:
    n, d = len(vecs), len(vecs[0])
    mean = [sum(v[j] for v in vecs) / n for j in range(d)]
    return [[v[j] - mean[j] for j in range(d)] for v in vecs]

def _power_iter(centered: list[list[float]], n_iter: int = 20) -> list[float]:
    """Нахождение первого собственного вектора (power iteration)."""
    d = len(centered[0])
    # начальный вектор — первая строка
    v = list(centered[0]) if any(abs(x) > 1e-10 for x in centered[0]) else [1.0] + [0.0]*(d-1)
    for _ in range(n_iter):
        # Av = X^T (X v)
        scores = [_dot(row, v) for row in centered]
        new_v  = [sum(scores[i] * centered[i][j] for i in range(len(centered))) for j in range(d)]
        norm   = math.sqrt(sum(x*x for x in new_v)) or 1.0
        v      = [x / norm for x in new_v]
    return v

def _pca2d(
    node_embeddings: dict[str, list[float]]
) -> dict[str, tuple[float, float]]:
    """
    Проецируем N-мерные эмбеддинги на 2 главные компоненты (PCA-lite).
    Если < 2 нод → возвращаем тривиальную проекцию.
    """
    ids  = list(node_embeddings.keys())
    vecs = [node_embeddings[i] for i in ids]
    d    = max(len(v) for v in vecs) if vecs else 0

    if len(ids) <= 1 or d < 2:
        # тривиальная проекция
        return {
            nid: (vecs[i][0] if len(vecs[i]) > 0 else 0.0,
                  vecs[i][1] if len(vecs[i]) > 1 else 0.0)
            for i, nid in enumerate(ids)
        }

    # выравниваем до одинаковой размерности
    vecs = [v + [0.0] * (d - len(v)) for v in vecs]

    # центрируем
    centered = _sub_mean(vecs)

    # первая компонента
    pc1 = _power_iter(centered)
    scores1 = [_dot(row, pc1) for row in centered]

    # вычитаем проекцию на pc1
    deflated = [
        [centered[i][j] - scores1[i] * pc1[j] for j in range(d)]
        for i in range(len(centered))
    ]

    # вторая компонента
    if any(any(abs(x) > 1e-10 for x in row) for row in deflated):
        pc2 = _power_iter(deflated)
        scores2 = [_dot(row, pc2) for row in deflated]
    else:
        scores2 = [0.0] * len(ids)

    return {nid: (scores1[i], scores2[i]) for i, nid in enumerate(ids)}


def _compute_modularity(km: "KnowledgeMap") -> float:
    """
    Модулярность Newman-Girvan Q ∈ (-0.5, 1.0).
    Q > 0.3 — хорошее разбиение, Q > 0.5 — отличное.

    Q = (1/2m) * Σ_ij [A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)

    A_ij  = вес ребра i→j
    k_i   = сумма весов рёбер из i (degree)
    m     = половина суммы всех весов
    δ     = 1 если i и j в одном сообществе
    """
    if not km.edges or not km.communities:
        return 0.0

    # карта нода → сообщество
    node_comm: dict[str, str] = {}
    for comm in km.communities.values():
        for n in comm.nodes:
            node_comm[n.id] = comm.id

    # строим взвешенный граф как словарь
    adj: dict[str, dict[str, float]] = {nid: {} for nid in km.nodes}
    m2 = 0.0
    for e in km.edges:
        w = e.weight
        adj[e.source][e.target] = adj[e.source].get(e.target, 0.0) + w
        adj[e.target][e.source] = adj[e.target].get(e.source, 0.0) + w
        m2 += 2 * w

    if m2 == 0.0:
        return 0.0

    # степени нод
    degree = {nid: sum(adj[nid].values()) for nid in km.nodes}

    Q = 0.0
    for e in km.edges:
        i, j, w = e.source, e.target, e.weight
        if node_comm.get(i) == node_comm.get(j):
            Q += w - degree[i] * degree[j] / m2
        # обратное ребро
        if node_comm.get(j) == node_comm.get(i):
            Q += w - degree[j] * degree[i] / m2

    return Q / m2


def _label_propagation(km: "KnowledgeMap", max_iter: int = 20) -> None:
    """
    Label Propagation поверх Q6 Voronoi-разбиения.

    Рафинирует принадлежность нод к сообществам по топологии графа:
    каждая нода итеративно «голосует» за сообщество своих соседей
    (взвешено по силе ребра). Нода переходит в новое сообщество,
    если соседи «голосуют» за другое сообщество сильнее, чем за текущее.

    Это устраняет ситуацию когда Q6 кладёт семантически далёкие ноды
    вместе только потому что у них совпал хеш.
    """
    if not km.edges:
        return

    # индекс: node_id → community_id
    node_to_comm: dict[str, str] = {}
    for comm in km.communities.values():
        for n in comm.nodes:
            node_to_comm[n.id] = comm.id

    # adjacency: node_id → list[(neighbor_id, weight)]
    adj: dict[str, list[tuple[str, float]]] = {nid: [] for nid in km.nodes}
    for e in km.edges:
        adj[e.source].append((e.target, e.weight))
        adj[e.target].append((e.source, e.weight))

    iters = 0
    import random
    rng = random.Random(42)

    for it in range(max_iter):
        changed = 0
        order = list(km.nodes.keys())
        rng.shuffle(order)

        for nid in order:
            nbrs = adj.get(nid, [])
            if not nbrs:
                continue
            votes: dict[str, float] = {}
            curr = node_to_comm[nid]
            # self-vote: сильный вес для стабильности (требует явного большинства чтобы переехать)
            votes[curr] = 1.0
            for nb_id, w in nbrs:
                nb_comm = node_to_comm.get(nb_id)
                if nb_comm:
                    votes[nb_comm] = votes.get(nb_comm, 0.0) + w
            best = max(votes, key=votes.__getitem__)
            # переезжаем только если чужое сообщество ЗНАЧИТЕЛЬНО сильнее (порог 1.5×)
            if best != curr and votes[best] > votes.get(curr, 0.0) * 1.5:
                node_to_comm[nid] = best
                changed += 1

        iters += 1
        if changed == 0:
            break

    km.lp_iterations = iters

    # перестраиваем сообщества по новым меткам
    new_groups: dict[str, list] = {}
    for nid, cid in node_to_comm.items():
        new_groups.setdefault(cid, []).append(km.nodes[nid])

    to_delete = []
    for cid, comm in list(km.communities.items()):
        new_nodes = new_groups.get(cid, [])
        if not new_nodes:
            to_delete.append(cid)
            continue
        if [n.id for n in new_nodes] == [n.id for n in comm.nodes]:
            continue   # без изменений
        comm.nodes = new_nodes
        comm.edges = [
            e for e in km.edges
            if e.source in {n.id for n in new_nodes}
            and e.target in {n.id for n in new_nodes}
        ]
        positions_2d = _pca2d({n.id: n.embedding for n in new_nodes})
        comm.build_all_signatures(node_2d_positions=positions_2d)

    # удаляем пустые сообщества
    for cid in to_delete:
        del km.communities[cid]


@dataclass
class KnowledgeMap:
    """
    Полная карта знаний — сота из сообществ.
    """
    nodes:       dict[str, GraphNode]     = field(default_factory=dict)
    edges:       list[GraphEdge]          = field(default_factory=list)
    hyper_edges:   list[HyperEdge]          = field(default_factory=list)
    communities:   dict[str, Community]     = field(default_factory=dict)
    borders:       list[CommunityBorder]    = field(default_factory=list)
    metadata:      dict[str, Any]           = field(default_factory=dict)
    lp_iterations: int                      = 0
    modularity:    float                    = 0.0
    pca_positions: dict[str, tuple[float, float]] = field(default_factory=dict)

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

            # 2D позиции через PCA-lite (первые 2 главные компоненты, power iteration)
            positions_2d = _pca2d({n.id: n.embedding for n in comm_nodes})

            community.build_all_signatures(node_2d_positions=positions_2d)
            self.add_community(community)

        # 3b. Label Propagation — рафинируем сообщества по топологии графа
        _label_propagation(self)

        # 3c. Модулярность Q — качество разбиения на сообщества
        self.modularity = _compute_modularity(self)

        # 4. Гиперрёбра
        all_positions = _pca2d({n.id: n.embedding for n in self.nodes.values()})
        self.pca_positions = all_positions   # сохраняем для ASCII/HTML
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
    n_points: int = 48,
) -> list[tuple[float, float]]:
    """
    Фрактальная граничная кривая между двумя сообществами.
    Сложность кривой зависит от:
      - Q6 Hamming distance (геометрическая дальность)
      - числа cross-edges (семантическая связанность)
    Используется многочастотное возмущение для нетривиального fd_box.
    """
    ea = ca.center_embedding
    eb = cb.center_embedding
    if not ea or not eb:
        return [(float(i) / (n_points - 1), 0.0) for i in range(n_points)]

    x0 = ea[0] if len(ea) > 0 else 0.0
    y0 = ea[1] if len(ea) > 1 else 0.0
    x1 = eb[0] if len(eb) > 0 else 1.0
    y1 = eb[1] if len(eb) > 1 else 0.0

    hd         = hamming(ca.hex_id, cb.hex_id)  # 1-6
    n_cross    = len(_shared_neighbors(ca, cb))  # общие ноды
    complexity = hd / 6.0 + n_cross * 0.15      # 0..1+

    # амплитуда базовая + бонус за сложность
    base_amp = 0.15 + complexity * 0.25

    dx = x1 - x0
    dy = y1 - y0
    length = math.hypot(dx, dy) or 1.0
    # перпендикулярный единичный вектор
    px = -dy / length
    py =  dx / length

    # детерминированный seed из id сообществ
    seed = sum(ord(c) for c in ca.id + cb.id)

    curve = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = x0 + t * dx
        y = y0 + t * dy

        # многочастотное возмущение — фрактальный суммарный шум
        noise = 0.0
        for harmonic in range(1, 5):
            freq  = harmonic * (1 + hd)
            phase = (seed * harmonic * 0.37) % (2 * math.pi)
            amp_h = base_amp / harmonic          # 1/f спектр
            noise += amp_h * math.sin(math.pi * t * freq + phase)

        # масштабируем на длину базовой линии
        noise *= length
        x += noise * px
        y += noise * py
        curve.append((x, y))

    return curve


def _shared_neighbors(ca: Community, cb: Community) -> list[str]:
    a_ids = {n.id for n in ca.nodes}
    b_ids = {n.id for n in cb.nodes}
    return list(a_ids & b_ids)
