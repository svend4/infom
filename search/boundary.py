"""
Boundary Search — поиск по фрактальным границам между сообществами.
Похожие IFS-коэффициенты → похожий характер отношений.
"""
from __future__ import annotations
from dataclasses import dataclass

from graph import KnowledgeMap, Community
from graph.community import CommunityBorder
from signatures import ifs_distance


@dataclass
class BoundarySearchResult:
    query_border:   CommunityBorder
    similar_borders: list[tuple[CommunityBorder, float]]  # (border, similarity)


def boundary_search(
    knowledge_map:  KnowledgeMap,
    community_a_id: str,
    community_b_id: str,
    top_k:          int = 5,
) -> BoundarySearchResult | None:
    """
    Найти похожие границы для пары сообществ.
    Similarity по IFS-коэффициентам FractalSignature.
    """
    target = next(
        (b for b in knowledge_map.borders
         if b.community_a == community_a_id and b.community_b == community_b_id
         or b.community_a == community_b_id and b.community_b == community_a_id),
        None
    )
    if not target:
        return None

    similar = knowledge_map.similar_boundaries(target, top_k=top_k)
    scored  = [(b, target.similarity(b)) for b in similar]
    scored.sort(key=lambda x: x[1], reverse=True)

    return BoundarySearchResult(
        query_border    = target,
        similar_borders = scored,
    )


def find_fuzzy_borders(
    knowledge_map: KnowledgeMap,
    threshold:     float = 1.5,
) -> list[CommunityBorder]:
    """
    Найти все нечёткие/размытые границы.
    fd_box > threshold означает сложную/перемешанную границу.
    """
    return [b for b in knowledge_map.borders
            if b.fractal.fd_box > threshold]


def find_sharp_borders(
    knowledge_map: KnowledgeMap,
    threshold:     float = 1.2,
) -> list[CommunityBorder]:
    """Найти все чёткие/разделённые границы."""
    return [b for b in knowledge_map.borders
            if b.fractal.fd_box < threshold]
