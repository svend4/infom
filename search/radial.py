"""
Radial Search — поиск в Q6-радиусе (Hamming ball).
Находит все сообщества в заданном семантическом радиусе.
"""
from __future__ import annotations
from dataclasses import dataclass

from graph import KnowledgeMap, Community, GraphNode
from signatures import hamming_ball, hamming, median, metric_interval, antipode


@dataclass
class RadialSearchResult:
    center_community: Community
    radius:           int
    communities:      list[Community]          # все в радиусе
    path_to_antipode: list[Community]          # путь к противоположному
    antipode_community: Community | None


def radial_search(
    knowledge_map: KnowledgeMap,
    community_id:  str,
    radius:        int = 1,
) -> RadialSearchResult | None:
    """
    Поиск в Q6-радиусе от сообщества.
    Использует структуру гиперкуба из meta/hexcore.
    """
    center = knowledge_map.communities.get(community_id)
    if not center:
        return None

    ball_ids     = set(hamming_ball(center.hex_id, radius))
    in_radius    = [c for c in knowledge_map.communities.values()
                    if c.hex_id in ball_ids and c.id != community_id]

    antipode_hex = center.hex_id ^ 63   # flip all 6 bits
    ant_comm     = next(
        (c for c in knowledge_map.communities.values()
         if c.hex_id == antipode_hex),
        None
    )

    path = []
    if ant_comm:
        path = knowledge_map.path_between(community_id, ant_comm.id)

    return RadialSearchResult(
        center_community   = center,
        radius             = radius,
        communities        = in_radius,
        path_to_antipode   = path,
        antipode_community = ant_comm,
    )


def semantic_neighbors(
    knowledge_map: KnowledgeMap,
    node_id:       str,
    top_k:         int = 6,
) -> list[GraphNode]:
    """
    Семантические соседи ноды по Q6 (ровно 6 как в гексаграмме).
    """
    node = knowledge_map.nodes.get(node_id)
    if not node:
        return []

    flip_ids = [node.hex_id ^ (1 << i) for i in range(6)]  # 6 соседей в Q6
    result   = []
    for n in knowledge_map.nodes.values():
        if n.id != node_id and n.hex_id in flip_ids:
            result.append(n)
    result.sort(key=lambda n: hamming(node.hex_id, n.hex_id))
    return result[:top_k]


def find_antipodal_community(
    knowledge_map: KnowledgeMap,
    community_id:  str,
) -> Community | None:
    """Найти 'противоположное' сообщество (антипод в Q6 — все 6 битов flip)."""
    center = knowledge_map.communities.get(community_id)
    if not center:
        return None
    ant_hex = center.hex_id ^ 63
    return next(
        (c for c in knowledge_map.communities.values() if c.hex_id == ant_hex),
        None,
    )
