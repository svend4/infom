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
            f"  Этап 1 (Hamming ball): {self.n_stage1} кандидатов",
            f"  Этап 2 (rerank):       {self.n_stage2} финальных",
            f"  Всего проверено:       {self.total_checked}",
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
    ):
        self.km            = km
        self.stage1_radius = stage1_radius
        self.stage2_top_k  = stage2_top_k

    # ── Этап 1: быстрый отбор через Q6 Hamming ball ──────────────────────────

    def _stage1_communities(self, query_hex: int) -> list[Community]:
        """Быстро найти кандидатов-сообществ в Q6-радиусе."""
        ball = set(hamming_ball(query_hex, self.stage1_radius))
        candidates = []
        for comm in self.km.communities.values():
            if comm.hex_id in ball:
                candidates.append(comm)
            # также добавляем соседей по Делоне (смежные сообщества)
            elif any(
                nb_id in {c.id for c in candidates}
                for nb_id in comm.neighbors
            ):
                candidates.append(comm)
        return candidates

    def _stage1_nodes(self, query_hex: int) -> list[GraphNode]:
        """Быстро найти ноды-кандидаты через Q6."""
        ball = set(hamming_ball(query_hex, self.stage1_radius))
        return [n for n in self.km.nodes.values() if n.hex_id in ball]

    # ── Этап 2: точный reranking по геометрической сигнатуре ─────────────────

    def _score_community(
        self, comm: Community, query_vec: list[float]
    ) -> float:
        """Полная geometric similarity: signature_vector cosine + Q6 bonus."""
        if not comm.signature_vector():
            return 0.0
        cv = comm.signature_vector()
        # косинусная схожесть
        n  = min(len(query_vec), len(cv))
        dot   = sum(query_vec[i] * cv[i] for i in range(n))
        norm_q = math.sqrt(sum(x*x for x in query_vec[:n])) or 1.0
        norm_c = math.sqrt(sum(x*x for x in cv[:n]))        or 1.0
        cosine = (dot / (norm_q * norm_c) + 1.0) / 2.0   # [0,1]
        return cosine

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

        # ── Этап 1: Q6 Hamming ball ───────────────────────────────────────────
        stage1_comms = self._stage1_communities(query_hex)
        stage1_nodes = self._stage1_nodes(query_hex) if include_nodes else []
        total_checked = len(self.km.communities) + len(self.km.nodes)

        # query_vec для geometric scoring = signature вектор из эмбеддинга
        # используем embedding как proxy (первые 6 компонент)
        query_vec = query_embedding[:6]

        for comm in stage1_comms:
            hd    = hamming(query_hex, comm.hex_id)
            score = self._score_community(comm, query_vec)
            # Q6-бонус: чем ближе, тем больше
            q6_bonus = 1.0 - hd / 7.0
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
            hd    = hamming(query_hex, node.hex_id)
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

        # HyperEdge matching по shape + nodes
        if include_hypers:
            for he in self.km.hyper_edges:
                # гиперребро релевантно если хотя бы одна его нода — кандидат
                node_ids  = set(he.nodes)
                cand_ids  = {c.item_id for c in top_for_rerank if c.item_type == "node"}
                overlap   = len(node_ids & cand_ids)
                if overlap > 0:
                    score = overlap / len(node_ids)
                    top_for_rerank.append(HNSWCandidate(
                        item_id   = he.id,
                        item_type = "hyper_edge",
                        score     = score,
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
