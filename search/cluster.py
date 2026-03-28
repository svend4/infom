"""
Cluster Search — поиск по геометрической форме кластера (TangramSignature).
Находит сообщества и гиперрёбра с заданным типом формы.
"""
from __future__ import annotations
from dataclasses import dataclass
import math

from graph import KnowledgeMap, Community, HyperEdge
from signatures import ShapeClass, TangramSignature


@dataclass
class ClusterSearchResult:
    shape:       ShapeClass
    communities: list[Community]
    hyper_edges: list[HyperEdge]
    total:       int


def cluster_search_by_shape(
    knowledge_map: KnowledgeMap,
    shape:         ShapeClass,
) -> ClusterSearchResult:
    """Найти все кластеры с заданным Tangram-shape."""
    comms = [c for c in knowledge_map.communities.values()
             if c.tangram and c.tangram.shape_class == shape]
    hes   = [he for he in knowledge_map.hyper_edges
             if he.shape == shape]
    return ClusterSearchResult(
        shape        = shape,
        communities  = comms,
        hyper_edges  = hes,
        total        = len(comms) + len(hes),
    )


def cluster_search_by_size(
    knowledge_map: KnowledgeMap,
    min_nodes:     int = 3,
    max_nodes:     int = 8,
) -> list[HyperEdge]:
    """Найти гиперрёбра по числу нод."""
    return [he for he in knowledge_map.hyper_edges
            if min_nodes <= he.n_nodes <= max_nodes]


def cluster_search_by_archetype(
    knowledge_map: KnowledgeMap,
    archetype:     str,
) -> list[Community]:
    """Найти сообщества доминирующего архетипа."""
    return [c for c in knowledge_map.communities.values()
            if any(n.archetype == archetype for n in c.nodes)]


def similar_clusters(
    knowledge_map: KnowledgeMap,
    community_id:  str,
    top_k:         int = 5,
) -> list[tuple[Community, float]]:
    """
    Найти сообщества похожие по геометрической сигнатуре.
    Возвращает [(community, distance)] отсортированные по близости.
    """
    target = knowledge_map.communities.get(community_id)
    if not target:
        return []

    results = []
    for comm in knowledge_map.communities.values():
        if comm.id == community_id:
            continue
        dist = target.distance_to(comm)
        results.append((comm, dist))

    results.sort(key=lambda x: x[1])
    return results[:top_k]
