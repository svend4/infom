"""
HNSW-style двухэтапный поиск в KnowledgeMap.

Идея из статьи https://habr.com/ru/articles/1011992/:
  Этап 1 (быстрый, приближённый):
      Hamming ball в Q6 → находим кандидатов за O(1) операций
      Аналог IVF/LSH — проверяем только ближайшие ячейки Voronoi

  Этап 2 (точный, медленный):
      Полная геометрическая similarity по signature_vector
      IFS reranking для границ
      Cross-encoder стиль: LLM оценивает топ-K

Иерархия уровней (аналог HNSW слоёв):
  L3  KnowledgeMap  — глобальный поиск по Q6 (Hamming ball)
  L2  Community     — поиск по signature_vector (Tangram+Fractal+Heptagram)
  L1  HyperEdge     — поиск по Tangram shape + IFS
  L0  GraphNode     — поиск по embedding similarity
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math

from graph import KnowledgeMap, Community, GraphNode, HyperEdge
from signatures import hamming_ball, hamming, ifs_distance
from search.multi_lsh import MultiProjectionQ6


# ── результаты ──────────────────────────────────────────────────────────────

@dataclass
class HNSWCandidate:
    item_id:   str
    item_type: str          # "node" | "community" | "hyper_edge"
    score:     float        # 0..1, выше = релевантнее
    distance:  float        # геометрическое расстояние
    stage:     int          # 1=быстрый, 2=точный


@dataclass
class HNSWResult:
    query_hex:    int
    candidates:   list[HNSWCandidate]
    n_stage1:     int        # кандидатов после этапа 1
    n_stage2:     int        # кандидатов после этапа 2 (rerank)
    total_checked: int

    @property
    def top(self) -> list[HNSWCandidate]:
        return self.candidates

    def summary(self) -> str:
        lines = [
            f"HNSW поиск: hex_id={self.query_hex}",
            f"  Этап 1 (Multi-Proj Q6): {self.n_stage1} кандидатов",
            f"  Этап 2 (rerank):        {self.n_stage2} финальных",
            f"  Всего проверено:        {self.total_checked}",
            "",
            "Топ результаты:",
        ]
        for c in self.candidates[:8]:
            bar = "█" * int(c.score * 12) + "░" * (12 - int(c.score * 12))
            lines.append(
                f"  [{c.item_type:10s}] {c.item_id[-12:]:12s}"
                f"  [{bar}] {c.score:.3f}  d={c.distance:.3f}"
            )
        return "\n".join(lines)


# ── основной класс ────────────────────────────────────────────────────────────

class HNSWSearch:
    """
    Двухэтапный иерархический поиск.

    stage1_radius:  Q6 Hamming radius для быстрого отбора (обычно 1-2)
    stage2_top_k:   сколько финальных результатов вернуть
    """

    def __init__(
        self,
        km:             KnowledgeMap,
        stage1_radius:  int = 2,
        stage2_top_k:   int = 10,
        n_projections:  int = 3,
    ):
        self.km            = km
        self.stage1_radius = stage1_radius
        self.stage2_top_k  = stage2_top_k
        self.multi_lsh     = MultiProjectionQ6(n_projections=n_projections)

    # ── Этап 1: быстрый отбор через Q6 Hamming ball ──────────────────────────

    def _stage1_communities(
        self, query_emb: list[float]
    ) -> tuple[list[Community], dict[str, float]]:
        """
        Быстро найти кандидатов-сообществ через union Hamming ball
        от k независимых Q6 проекций (Multi-Projection LSH).

        Возвращает (candidates, coverage) где coverage[comm_id] ∈ [0,1] —
        доля проекций, подтверждающих близость (score boost в этапе 2).
        """
        union_ball = self.multi_lsh.union_hamming_ball(
            query_emb, self.stage1_radius
        )
        candidates  = []
        coverage    = {}
        already_ids: set[str] = set()

        for comm in self.km.communities.values():
            if comm.hex_id in union_ball:
                cov = self.multi_lsh.coverage_score(
                    query_emb, comm.hex_id, self.stage1_radius
                )
                candidates.append(comm)
                coverage[comm.id] = cov
                already_ids.add(comm.id)

        # Делоне-соседи уже найденных сообществ (граничные кандидаты)
        for comm in list(candidates):
            for nb_id in comm.neighbors:
                if nb_id not in already_ids:
                    nb = self.km.communities.get(nb_id)
                    if nb:
                        candidates.append(nb)
                        coverage[nb.id] = 0.1  # слабый coverage
                        already_ids.add(nb_id)

        return candidates, coverage

    def _stage1_nodes(self, query_emb: list[float]) -> list[GraphNode]:
        """Быстро найти ноды-кандидаты через multi-projection union."""
        union_ball = self.multi_lsh.union_hamming_ball(
            query_emb, self.stage1_radius
        )
        return [n for n in self.km.nodes.values() if n.hex_id in union_ball]

    # ── Этап 2: точный reranking по геометрической сигнатуре ─────────────────

    # архетипы по ключевым словам домена (для буста)
    _DOMAIN_ARCH = {
        "technology": {"ADCO", "ADEO", "MDCO"},
        "science":    {"ASCO", "ASEO", "ADEO"},
        "biology":    {"MDEF", "MSCF", "MSEO"},
        "urbanism":   {"MDCF", "MSCO", "MDCO"},
    }

    def _archetype_boost(
        self, comm: Community, query: str
    ) -> float:
        """
        Буст сообщества если его доминирующий архетип совпадает с доменом запроса.
        Возвращает 0.0..0.15 (скромная добавка, не перебивает geometric score).
        """
        q = query.lower()
        dom_arch = comm.dominant_archetype
        for domain, archs in self._DOMAIN_ARCH.items():
            # простая проверка по ключевым словам (без импорта QueryExpander)
            kws = {
                "technology": ["алгоритм","нейросет","компил","программ","код","систем"],
                "science":    ["физик","математ","термо","теори","закон","наук"],
                "biology":    ["клетк","днк","ген","организм","экосис","биол"],
                "urbanism":   ["метро","транспорт","инфраструктур","город","улиц"],
            }
            if any(kw in q for kw in kws.get(domain, [])):
                if dom_arch in archs:
                    return 0.15
                break
        return 0.0

    def _score_community(
        self, comm: Community, query_vec: list[float], query_text: str = ""
    ) -> float:
        """Geometric similarity: signature_vector cosine + архетип-буст."""
        if not comm.signature_vector():
            return 0.0
        cv = comm.signature_vector()
        n  = min(len(query_vec), len(cv))
        dot    = sum(query_vec[i] * cv[i] for i in range(n))
        norm_q = math.sqrt(sum(x*x for x in query_vec[:n])) or 1.0
        norm_c = math.sqrt(sum(x*x for x in cv[:n]))        or 1.0
        cosine = (dot / (norm_q * norm_c) + 1.0) / 2.0
        boost  = self._archetype_boost(comm, query_text) if query_text else 0.0
        return min(1.0, cosine + boost)

    def _score_node(
        self, node: GraphNode, query_emb: list[float]
    ) -> float:
        """Embedding cosine similarity."""
        n   = min(len(query_emb), len(node.embedding))
        if n == 0:
            return 0.0
        dot   = sum(query_emb[i] * node.embedding[i] for i in range(n))
        na    = math.sqrt(sum(x*x for x in query_emb[:n]))    or 1.0
        nb    = math.sqrt(sum(x*x for x in node.embedding[:n])) or 1.0
        return (dot / (na * nb) + 1.0) / 2.0

    # ── IFS reranker для границ ────────────────────────────────────────────────

    def _ifs_rerank(
        self, candidates: list[HNSWCandidate], query_ifs: list[float]
    ) -> list[HNSWCandidate]:
        """
        Reranker: уточняем score границ через IFS-расстояние.
        Аналог cross-encoder из статьи — медленно, но точно.
        """
        for c in candidates:
            if c.item_type != "community":
                continue
            comm = self.km.communities.get(c.item_id)
            if not comm or not comm.fractal:
                continue
            d_ifs = ifs_distance(query_ifs, comm.fractal.ifs_coeffs)
            # чем ближе IFS — тем выше score
            ifs_bonus = 1.0 / (1.0 + d_ifs * 2)
            c.score   = 0.7 * c.score + 0.3 * ifs_bonus
        return candidates

    # ── полный поиск ──────────────────────────────────────────────────────────

    def search(
        self,
        query_embedding: list[float],
        query_ifs:       list[float] | None = None,
        include_nodes:   bool = True,
        include_hypers:  bool = True,
        query_text:      str  = "",
    ) -> HNSWResult:
        """
        Двухэтапный поиск:
          1. Hamming ball → список кандидатов
          2. Geometric rerank → топ-K

        query_embedding: 6D вектор (как у GraphNode)
        query_ifs:       IFS коэффициенты (опционально, для reranking границ)
        """
        from signatures import embed_to_q6
        query_hex = embed_to_q6(query_embedding)
        total_checked = 0

        all_candidates: list[HNSWCandidate] = []

        # ── Этап 1: Multi-Projection Q6 union ────────────────────────────────
        stage1_comms, coverage = self._stage1_communities(query_embedding)
        stage1_nodes = self._stage1_nodes(query_embedding) if include_nodes else []
        total_checked = len(self.km.communities) + len(self.km.nodes)

        # query_vec для geometric scoring
        query_vec = query_embedding[:6]

        for comm in stage1_comms:
            hd    = self.multi_lsh.min_distance(query_embedding, [0.0] * 6)
            hd    = hamming(query_hex, comm.hex_id)   # single-proj distance
            score = self._score_community(comm, query_vec, query_text)
            # Q6-бонус: coverage = доля проекций подтверждающих близость
            cov      = coverage.get(comm.id, 0.0)
            q6_bonus = (1.0 - hd / 7.0) * 0.7 + cov * 0.3
            score    = 0.6 * score + 0.4 * q6_bonus
            all_candidates.append(HNSWCandidate(
                item_id   = comm.id,
                item_type = "community",
                score     = score,
                distance  = float(hd),
                stage     = 1,
            ))

        for node in stage1_nodes:
            score = self._score_node(node, query_embedding)
            hd    = self.multi_lsh.min_distance(query_embedding, node.embedding)
            all_candidates.append(HNSWCandidate(
                item_id   = node.id,
                item_type = "node",
                score     = score,
                distance  = float(hd),
                stage     = 1,
            ))

        n_stage1 = len(all_candidates)

        # ── Этап 2: reranking ─────────────────────────────────────────────────
        # сортируем по score этапа 1, берём топ для точного rerank
        all_candidates.sort(key=lambda c: c.score, reverse=True)
        top_for_rerank = all_candidates[: self.stage2_top_k * 2]

        # IFS reranker (если передан query_ifs)
        if query_ifs:
            top_for_rerank = self._ifs_rerank(top_for_rerank, query_ifs)

        # HyperEdge matching: score = средняя embedding-similarity нод + overlap bonus
        if include_hypers:
            node_scores = {
                c.item_id: c.score
                for c in top_for_rerank
                if c.item_type == "node"
            }
            for he in self.km.hyper_edges:
                node_ids = set(he.nodes)
                matched  = node_ids & set(node_scores.keys())
                if not matched:
                    continue
                # средний score найденных нод (не просто overlap/total)
                avg_node_score = sum(node_scores[nid] for nid in matched) / len(node_ids)
                # штраф за частичное покрытие: 70% weight если не все ноды найдены
                coverage_ratio = len(matched) / len(node_ids)
                he_score = avg_node_score * (0.7 + 0.3 * coverage_ratio)
                top_for_rerank.append(HNSWCandidate(
                    item_id   = he.id,
                    item_type = "hyper_edge",
                    score     = he_score,
                    distance  = 0.0,
                    stage     = 2,
                ))

        # финальная сортировка
        top_for_rerank.sort(key=lambda c: c.score, reverse=True)
        final = top_for_rerank[: self.stage2_top_k]

        return HNSWResult(
            query_hex     = query_hex,
            candidates    = final,
            n_stage1      = n_stage1,
            n_stage2      = len(final),
            total_checked = total_checked,
        )

    # ── удобные методы ────────────────────────────────────────────────────────

    def search_by_text_embedding(
        self, node_label: str
    ) -> HNSWResult:
        """
        Поиск по метке ноды: находим ноду, используем её embedding.
        """
        node = next(
            (n for n in self.km.nodes.values()
             if node_label.lower() in n.label.lower()),
            None,
        )
        if node:
            query_ifs = None
            comm = self.km.find_community(node.id)
            if comm and comm.fractal:
                query_ifs = comm.fractal.ifs_coeffs
            return self.search(node.embedding, query_ifs=query_ifs)
        # запасной: используем нулевой вектор
        return self.search([0.0] * 6)
