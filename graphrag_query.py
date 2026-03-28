"""
GraphRAG Query — полный retrieve+generate pipeline.

Этапы:
  1. Embed  — получаем embedding запроса (LLM или Mock)
  2. HNSW   — двухэтапный поиск: Hamming ball → geometric rerank
  3. Context — собираем контекст из найденных нод/сообществ/гиперрёбер
  4. Expand  — pseudorag QueryExpander → структурированные вопросы
  5. Generate — LLM генерирует ответ с контекстом

Режимы (аналог GraphRAG):
  local   — поиск вокруг конкретной ноды
  global  — MapReduce по всем сообществам
  hybrid  — local + global + HNSW rerank
"""
from __future__ import annotations
from dataclasses import dataclass, field

from graph         import KnowledgeMap
from search.hnsw   import HNSWSearch, HNSWResult
from archetypes    import QueryExpander, QuestionTree
from llm_adapter   import LLMAdapter, MockLLMAdapter


# ── промпты ─────────────────────────────────────────────────────────────────

LOCAL_PROMPT = """Ты — эксперт по анализу знаний. Ответь на вопрос, используя ТОЛЬКО предоставленный контекст.

Вопрос: {query}

Контекст из графа знаний:
{context}

Геометрический профиль (информация о кластере):
  Форма кластера: {shape}
  Скелет: {skeleton}
  Доминирующее измерение: {dominant_dim}

Уточняющие вопросы для глубокого ответа:
{expansion_questions}

Дай краткий, точный ответ основываясь только на контексте."""

GLOBAL_PROMPT = """Ты — аналитик. Синтезируй ответ на вопрос по ВСЕМ сообществам графа знаний.

Вопрос: {query}

Сводка по сообществам:
{community_summaries}

Дай структурированный ответ с выводами по каждой группе знаний."""

CONTEXT_TEMPLATE = """Сущности: {entities}
Связи: {relations}
Архетипы: {archetypes}
Q6-позиция: {hex_id} [{bits}]
Форма кластера: {shape} / Скелет: {skeleton}"""


# ── структуры ────────────────────────────────────────────────────────────────

@dataclass
class GraphRAGAnswer:
    query:       str
    mode:        str
    answer:      str
    context:     str
    hnsw_result: HNSWResult | None      = None
    questions:   QuestionTree | None    = None
    sources:     list[str]              = field(default_factory=list)
    tokens_used: int                    = 0

    def display(self) -> str:
        lines = [
            f"{'='*60}",
            f"Запрос [{self.mode}]: {self.query}",
            f"{'='*60}",
            f"\nОтвет:\n{self.answer}",
            f"\n{'─'*60}",
            f"Источники: {', '.join(self.sources[:5])}",
            f"Токены: {self.tokens_used}",
        ]
        if self.questions:
            hi_pri = self.questions.by_priority(min_p=4)[:3]
            if hi_pri:
                lines.append(f"\nКлючевые вопросы [QueryExpander]:")
                for q in hi_pri:
                    lines.append(f"  [{q.archetype}] {q.text}")
        if self.hnsw_result:
            lines.append(f"\n{self.hnsw_result.summary()}")
        return "\n".join(lines)


# ── главный класс ─────────────────────────────────────────────────────────────

class GraphRAGQuery:
    """
    Полный RAG pipeline над KnowledgeMap.

    Использование:
        rag = GraphRAGQuery(km, llm=OllamaAdapter())
        answer = rag.query("Что такое нейросеть?", mode="hybrid")
        print(answer.display())
    """

    def __init__(
        self,
        km:              KnowledgeMap,
        llm:             LLMAdapter | None  = None,
        stage1_radius:   int = 2,
        top_k:           int = 5,
    ):
        self.km       = km
        self.llm      = llm or MockLLMAdapter()
        self.hnsw     = HNSWSearch(km, stage1_radius=stage1_radius, stage2_top_k=top_k)
        self.expander = QueryExpander()
        self.top_k    = top_k

    # ── режим LOCAL ──────────────────────────────────────────────────────────

    def _local_query(self, query: str, query_emb: list[float]) -> GraphRAGAnswer:
        """Поиск вокруг наиболее релевантной ноды."""
        hnsw_result = self.hnsw.search(query_emb)

        # собираем ноды из топ-результатов
        top_nodes = [
            self.km.nodes[c.item_id]
            for c in hnsw_result.candidates
            if c.item_type == "node" and c.item_id in self.km.nodes
        ][:self.top_k]

        top_comms = [
            self.km.communities[c.item_id]
            for c in hnsw_result.candidates
            if c.item_type == "community" and c.item_id in self.km.communities
        ][:3]

        # контекст
        entities  = ", ".join(n.label for n in top_nodes) or "нет данных"
        relations = self._extract_relations(top_nodes)
        archetypes_str = ", ".join(set(n.archetype for n in top_nodes if n.archetype))

        shape     = top_comms[0].tangram.shape_class.value if top_comms and top_comms[0].tangram else "unknown"
        skeleton  = top_comms[0].octagram.skeleton_type.value if top_comms and top_comms[0].octagram else "unknown"
        dom_dim   = (top_comms[0].heptagram.dominant_ray.label
                     if top_comms and top_comms[0].heptagram else "unknown")

        context = CONTEXT_TEMPLATE.format(
            entities   = entities,
            relations  = relations,
            archetypes = archetypes_str,
            hex_id     = top_comms[0].hex_id if top_comms else "?",
            bits       = "".join(str(b) for b in top_comms[0].hex_sig.bits)
                         if top_comms and top_comms[0].hex_sig else "??????",
            shape      = shape,
            skeleton   = skeleton,
        )

        # QueryExpander: выбираем высокоприоритетные вопросы для усиления контекста
        qtree = self.expander.expand_query(query)
        hi_q  = qtree.by_priority(min_p=4)[:5]
        expansion_str = "\n".join(f"  - {q.text}" for q in hi_q) or "  (нет)"

        prompt = LOCAL_PROMPT.format(
            query               = query,
            context             = context,
            shape               = shape,
            skeleton            = skeleton,
            dominant_dim        = dom_dim,
            expansion_questions = expansion_str,
        )

        resp    = self.llm.complete(prompt)
        sources = [n.label for n in top_nodes]

        return GraphRAGAnswer(
            query       = query,
            mode        = "local",
            answer      = resp.text,
            context     = context,
            hnsw_result = hnsw_result,
            questions   = qtree,
            sources     = sources,
            tokens_used = resp.tokens,
        )

    # ── режим GLOBAL ─────────────────────────────────────────────────────────

    def _global_query(self, query: str) -> GraphRAGAnswer:
        """MapReduce по всем сообществам."""
        comm_summaries = []
        all_sources    = []

        for comm in self.km.communities.values():
            nodes_str = ", ".join(n.label for n in comm.nodes[:5])
            shape     = comm.tangram.shape_class.value if comm.tangram else "?"
            fd        = comm.fractal.fd_box if comm.fractal else 1.0
            summary   = (f"[{shape}] Q6={comm.hex_id}: {nodes_str} "
                        f"(fd_boundary={fd:.2f})")
            comm_summaries.append(summary)
            all_sources.extend(n.label for n in comm.nodes[:3])

        context = "\n".join(f"  {s}" for s in comm_summaries)
        prompt  = GLOBAL_PROMPT.format(
            query              = query,
            community_summaries = context,
        )

        resp = self.llm.complete(prompt)

        return GraphRAGAnswer(
            query       = query,
            mode        = "global",
            answer      = resp.text,
            context     = context,
            questions   = self.expander.expand_query(query),
            sources     = list(set(all_sources))[:10],
            tokens_used = resp.tokens,
        )

    # ── режим HYBRID ─────────────────────────────────────────────────────────

    def _hybrid_query(self, query: str, query_emb: list[float]) -> GraphRAGAnswer:
        """Local + Global + HNSW rerank."""
        local_ans  = self._local_query(query, query_emb)
        global_ans = self._global_query(query)

        # объединяем контексты
        combined_context = (
            "=== Local context ===\n" + local_ans.context +
            "\n\n=== Global context ===\n" + global_ans.context[:500]
        )

        combined_prompt = f"""Вопрос: {query}

{combined_context}

Синтезируй ответ используя оба контекста — локальный (конкретные сущности) и глобальный (все сообщества)."""

        resp = self.llm.complete(combined_prompt)

        return GraphRAGAnswer(
            query       = query,
            mode        = "hybrid",
            answer      = resp.text,
            context     = combined_context,
            hnsw_result = local_ans.hnsw_result,
            questions   = local_ans.questions,
            sources     = list(set(local_ans.sources + global_ans.sources))[:10],
            tokens_used = resp.tokens + local_ans.tokens_used + global_ans.tokens_used,
        )

    # ── публичный API ─────────────────────────────────────────────────────────

    def query(
        self,
        query_text:      str,
        mode:            str = "hybrid",
        query_embedding: list[float] | None = None,
    ) -> GraphRAGAnswer:
        """
        Обработать запрос.

        mode:
          "local"   — поиск вокруг ближайших нод
          "global"  — обзор всех сообществ
          "hybrid"  — local + global (рекомендуется)
        """
        if query_embedding is None:
            query_embedding = self.llm.embed(query_text)
            # дополняем до 6D если нужно
            while len(query_embedding) < 6:
                query_embedding.append(0.0)
            query_embedding = query_embedding[:6]

        if mode == "local":
            return self._local_query(query_text, query_embedding)
        elif mode == "global":
            return self._global_query(query_text)
        else:
            return self._hybrid_query(query_text, query_embedding)

    # ── вспомогательные ───────────────────────────────────────────────────────

    def _extract_relations(self, nodes: list) -> str:
        if not nodes:
            return "нет связей"
        node_ids  = {n.id for n in nodes}
        id_to_lbl = {n.id: n.label for n in self.km.nodes.values()}
        rels = [
            f"{id_to_lbl.get(e.source, e.source)} —{e.label}→ {id_to_lbl.get(e.target, e.target)}"
            for e in self.km.edges
            if e.source in node_ids or e.target in node_ids
        ][:8]
        return "; ".join(rels) if rels else "нет связей"
