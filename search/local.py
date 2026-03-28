"""
Local Search — поиск вокруг конкретной ноды или сущности.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from graph import GraphNode, GraphEdge, KnowledgeMap
from signatures import hamming


@dataclass
class LocalSearchResult:
    node:       GraphNode
    neighbors:  list[GraphNode]
    edges:      list[GraphEdge]
    community_id: str
    score:      float


def local_search(
    knowledge_map: KnowledgeMap,
    node_id:       str,
    depth:         int = 2,
    top_k:         int = 10,
) -> LocalSearchResult | None:
    """
    Поиск в локальном окружении ноды.
    Возвращает соседей в пределах depth шагов.
    """
    node = knowledge_map.nodes.get(node_id)
    if not node:
        return None

    # BFS по рёбрам
    visited = {node_id}
    frontier = [node_id]
    for _ in range(depth):
        next_f = []
        for nid in frontier:
            for edge in knowledge_map.edges:
                if edge.source == nid and edge.target not in visited:
                    visited.add(edge.target)
                    next_f.append(edge.target)
                elif not edge.directed and edge.target == nid and edge.source not in visited:
                    visited.add(edge.source)
                    next_f.append(edge.source)
        frontier = next_f

    neighbor_ids = visited - {node_id}
    neighbors    = [knowledge_map.nodes[nid] for nid in neighbor_ids
                    if nid in knowledge_map.nodes]

    # ранжирование по Q6-расстоянию
    neighbors.sort(key=lambda n: hamming(node.hex_id, n.hex_id))
    neighbors = neighbors[:top_k]

    relevant_edges = [
        e for e in knowledge_map.edges
        if e.source == node_id or e.target == node_id
    ]

    community    = knowledge_map.find_community(node_id)
    community_id = community.id if community else ""

    score = sum(e.weight for e in relevant_edges) / max(len(relevant_edges), 1)

    return LocalSearchResult(
        node         = node,
        neighbors    = neighbors,
        edges        = relevant_edges,
        community_id = community_id,
        score        = score,
    )
