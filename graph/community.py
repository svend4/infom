"""
Community — шестиугольное сообщество в графе знаний.
Несёт полный стек геометрических подписей:
    TangramSignature  (форма внутренней структуры)
    FractalSignature  (граница с соседями)
    HexSignature      (позиция в Q6)
    HeptagramSignature (многомерное описание отношений)
    OctagramSignature  (3D компас / скелет)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from signatures import (
    TangramSignature, build_tangram_signature,
    FractalSignature, build_fractal_signature,
    HexSignature,     build_hex_signature,
    HeptagramSignature, build_heptagram_signature,
    OctagramSignature,  build_octagram_signature, SkeletonType,
    ifs_distance, hamming, packing_number, voronoi_cells,
)
from graph.node    import GraphNode, GraphEdge
from graph.hyper_edge import HyperEdge


@dataclass
class CommunityBorder:
    """Граница между двумя сообществами (описана FractalSignature)."""
    community_a: str
    community_b: str
    fractal:     FractalSignature
    shared_nodes: list[str] = field(default_factory=list)

    @property
    def complexity(self) -> float:
        """Фрактальная размерность границы: 1=чёткая, 2=размытая."""
        return self.fractal.fd_box

    def similarity(self, other: "CommunityBorder") -> float:
        """Похожесть двух границ по IFS-коэффициентам."""
        return 1.0 / (1.0 + ifs_distance(
            self.fractal.ifs_coeffs, other.fractal.ifs_coeffs
        ))


@dataclass
class Community:
    """
    Шестиугольное сообщество — ячейка соты графа знаний.
    Соответствует Q6-ноде + Voronoi-ячейке.
    """
    id:        str
    label:     str
    nodes:     list[GraphNode]
    edges:     list[GraphEdge]       = field(default_factory=list)
    hyper_edges: list[HyperEdge]     = field(default_factory=list)

    # геометрические подписи
    tangram:   TangramSignature  | None = None
    fractal:   FractalSignature  | None = None
    hex_sig:   HexSignature      | None = None
    heptagram: HeptagramSignature | None = None
    octagram:  OctagramSignature  | None = None

    # связи
    neighbors: list[str]  = field(default_factory=list)  # id соседних сообществ
    borders:   list[CommunityBorder] = field(default_factory=list)

    metadata:  dict[str, Any] = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def center_embedding(self) -> list[float]:
        """Средний эмбеддинг всех нод сообщества."""
        if not self.nodes:
            return []
        dim = len(self.nodes[0].embedding)
        return [
            sum(n.embedding[i] for n in self.nodes) / len(self.nodes)
            for i in range(dim)
        ]

    @property
    def hex_id(self) -> int:
        return self.hex_sig.hex_id if self.hex_sig else 0

    def build_all_signatures(
        self,
        node_2d_positions: dict[str, tuple[float, float]] | None = None,
        boundary_curve:    list[tuple[float, float]] | None = None,
        relation_weights:  dict[str, float] | None = None,
        direction_weights: dict[str, float] | None = None,
    ) -> None:
        """Вычислить все геометрические подписи сообщества."""
        # TangramSignature — форма внутреннего графа
        if node_2d_positions:
            positions = list(node_2d_positions.values())
            if len(positions) >= 3:
                self.tangram = build_tangram_signature(positions)
            elif len(positions) == 2:
                # 2 ноды → отрезок: TangramSignature с ShapeClass.POLYGON
                from signatures import ShapeClass, TangramSignature
                self.tangram = TangramSignature(
                    polygon=positions, shape_class=ShapeClass.RECTANGLE,
                    centroid=((positions[0][0]+positions[1][0])/2,
                              (positions[0][1]+positions[1][1])/2),
                    angle=0.0, scale=1.0, area=0.0,
                )
            elif len(positions) == 1:
                from signatures import ShapeClass, TangramSignature
                self.tangram = TangramSignature(
                    polygon=positions, shape_class=ShapeClass.POLYGON,
                    centroid=positions[0], angle=0.0, scale=0.0, area=0.0,
                )

        # FractalSignature — граница сообщества
        if boundary_curve and len(boundary_curve) >= 4:
            self.fractal = build_fractal_signature(boundary_curve)

        # HexSignature — Q6 позиция
        center_emb = self.center_embedding
        if center_emb:
            self.hex_sig = build_hex_signature(center_emb)

        # HeptagramSignature — 7D отношений
        if relation_weights:
            self.heptagram = build_heptagram_signature(relation_weights)
        else:
            self.heptagram = _auto_heptagram(self.edges)

        # OctagramSignature — 3D компас
        if direction_weights:
            self.octagram = build_octagram_signature(direction_weights)
        else:
            self.octagram = _auto_octagram(self.nodes, self.hex_sig)

    def signature_vector(self) -> list[float]:
        """Единый вектор всех подписей для сравнения сообществ."""
        vec = []
        if self.hex_sig:
            vec += list(self.hex_sig.bits)
        if self.tangram:
            vec += [self.tangram.area, self.tangram.angle,
                    float(self.tangram.n_vertices)]
        if self.fractal:
            vec += [self.fractal.fd_box, self.fractal.fd_divider]
            vec += self.fractal.ifs_coeffs[:4]
        if self.heptagram:
            vec += [r.length for r in self.heptagram.rays]
        if self.octagram:
            vec += [r.length for r in self.octagram.rays]
        return vec

    def distance_to(self, other: "Community") -> float:
        """Геометрическое расстояние между двумя сообществами."""
        import math
        va = self.signature_vector()
        vb = other.signature_vector()
        n  = min(len(va), len(vb))
        if n == 0:
            return float('inf')
        return math.sqrt(sum((va[i]-vb[i])**2 for i in range(n)))

    def hamming_distance_to(self, other: "Community") -> int:
        """Q6 расстояние между двумя сообществами."""
        return hamming(self.hex_id, other.hex_id)


# ── автоматические подписи ───────────────────────────────────────────────────

def _auto_heptagram(edges: list[GraphEdge]) -> HeptagramSignature:
    if not edges:
        return build_heptagram_signature({})
    weights    = [e.weight for e in edges]
    mean_w     = sum(weights) / len(weights)
    directed_n = sum(1 for e in edges if e.directed)
    return build_heptagram_signature({
        "strength":   mean_w,
        "direction":  directed_n / max(len(edges), 1),
        "temporal":   0.5,
        "confidence": mean_w,
        "scale":      min(1.0, len(edges) / 20),
        "context":    max(weights) if weights else 0.5,
        "source":     0.7,
    })


def _auto_octagram(
    nodes: list[GraphNode],
    hex_sig: HexSignature | None,
) -> OctagramSignature:
    if hex_sig is None:
        return build_octagram_signature({d: 0.5 for d in
                                         ["N","NE","E","SE","S","SW","W","NW"]})
    bits = hex_sig.bits
    # Маппинг 6 битов Q6 → 8 направлений компаса
    dirs = {
        "N":  float(bits[0]),   # material → абстрактный/конкретный
        "NE": float(bits[1]),   # dynamic
        "E":  float(bits[2]),   # complex
        "SE": float(bits[3]),   # ordered
        "S":  1.0 - float(bits[0]),
        "SW": float(bits[4]),   # local
        "W":  float(bits[5]),   # explicit
        "NW": float(bits[1]) * float(bits[2]),
    }
    return build_octagram_signature(dirs)
