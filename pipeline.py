"""
InfoM Pipeline — главный оркестратор.

Архитектура:
    L0  GraphNode       — сущности с эмбеддингами
    L1  HyperEdge       — кластеры 3-8 нод (Tangram + Heptagram)
    L2  Community       — шестиугольные сообщества (Q6 + Fractal + Octagram)
    L3  KnowledgeMap    — полная сота сообществ

Геометрическая иерархия форм:
    △  Треугольник  (3)   — базовая триада           [meta2/Tangram]
    □  Прямоугольник(4)   — устойчивый паттерн       [meta2/Tangram]
    ⬠  Пятиугольник (5)   — сложный кластер          [meta2/Tangram]
    ⬡  Шестиугольник(6)   — Q6-сообщество/сота       [meta/Q6]
    ☆  Семилучевая  (7)   — 3D асимм. звезда магов   [новое]
    ✳  Восьмилучевая(8)   — 3D роза ветров/скелет    [новое]
    ∿  Фрактал            — граница между группами   [meta2/Fractal]
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import math

from graph          import GraphNode, GraphEdge, KnowledgeMap
from archetypes     import QueryExpander, QuestionTree
from search         import (
    local_search,  LocalSearchResult,
    cluster_search_by_shape, similar_clusters,
    boundary_search,
    radial_search,
    HNSWSearch,
)
from signatures     import ShapeClass
from llm_adapter    import LLMAdapter, MockLLMAdapter


@dataclass
class IndexConfig:
    n_communities:    int   = 8      # число шестиугольных сообществ
    min_cluster_size: int   = 3      # мин. размер гиперребра
    embedding_dim:    int   = 6      # размерность (по умолч. = N_DIMS Q6)


@dataclass
class QueryResult:
    query:        str
    questions:    QuestionTree
    local:        LocalSearchResult  | None = None
    communities:  list[Any]                 = field(default_factory=list)
    hyper_edges:  list[Any]                 = field(default_factory=list)
    summary:      str                       = ""


class InfoMPipeline:
    """
    Главный класс системы InfoM.

    Использование:
        pipeline = InfoMPipeline()

        # Индексация документа
        pipeline.index_document("Текст документа...")

        # Запрос
        result = pipeline.query("Что такое алгоритм?")
        print(result.summary)
    """

    def __init__(
        self,
        config: IndexConfig | None = None,
        llm:    LLMAdapter  | None = None,
    ):
        self.config   = config or IndexConfig()
        self.km       = KnowledgeMap()
        self.expander = QueryExpander()
        self.llm      = llm or MockLLMAdapter()
        self._rag: "GraphRAGQuery | None" = None  # lazy init

    # ── индексация ────────────────────────────────────────────────────────

    def add_node(
        self,
        node_id:   str,
        label:     str,
        embedding: list[float],
        archetype: str = "",
        metadata:  dict | None = None,
    ) -> GraphNode:
        """Добавить сущность в граф."""
        node = GraphNode(
            id        = node_id,
            label     = label,
            embedding = embedding,
            archetype = archetype,
            metadata  = metadata or {},
        )
        self.km.add_node(node)
        return node

    def add_edge(
        self,
        source:   str,
        target:   str,
        label:    str,
        weight:   float = 1.0,
        directed: bool  = True,
    ) -> GraphEdge:
        """Добавить связь между сущностями."""
        edge = GraphEdge(
            source   = source,
            target   = target,
            label    = label,
            weight   = weight,
            directed = directed,
        )
        self.km.add_edge(edge)
        return edge

    def build(self) -> None:
        """
        Построить граф знаний:
        - Voronoi разбиение → шестиугольные сообщества
        - Tangram-кластеры (гиперрёбра)
        - Fractal-границы
        - Граф Делоне между сообществами
        """
        self.km.build(n_communities=self.config.n_communities)

    # ── поиск ────────────────────────────────────────────────────────────

    def query(self, text: str, node_id: str | None = None) -> QueryResult:
        """
        Обработать запрос:
        1. pseudorag: expand → QuestionTree (16 архетипов + 7 Heptagram)
        2. Поиск в KnowledgeMap
        3. Ранжирование результатов
        """
        questions = self.expander.expand_query(text)

        # локальный поиск по ноде
        local = None
        if node_id:
            local = local_search(self.km, node_id)

        # найти релевантные сообщества
        communities = list(self.km.communities.values())

        # найти релевантные гиперрёбра
        hyper_edges = [
            he for he in self.km.hyper_edges
            if any(text.lower() in n.label.lower()
                   for n in self.km.nodes.values()
                   if n.id in he.nodes)
        ]

        result = QueryResult(
            query       = text,
            questions   = questions,
            local       = local,
            communities = communities[:10],
            hyper_edges = hyper_edges[:10],
        )
        result.summary = self._format_summary(result)
        return result

    def search_by_shape(self, shape: ShapeClass):
        """Поиск кластеров по геометрической форме (Tangram)."""
        return cluster_search_by_shape(self.km, shape)

    def search_radial(self, community_id: str, radius: int = 1):
        """Радиальный поиск в Q6 (Hamming ball)."""
        return radial_search(self.km, community_id, radius)

    def search_hnsw(self, query_embedding: list[float], radius: int = 2):
        """Двухэтапный HNSW поиск: Hamming ball → geometric rerank."""
        return HNSWSearch(self.km, stage1_radius=radius).search(query_embedding)

    def rag_query(self, text: str, mode: str = "hybrid") -> "GraphRAGAnswer":
        """
        Полный GraphRAG запрос: HNSW retrieve → LLM generate.
        mode: "local" | "global" | "hybrid"
        """
        from graphrag_query import GraphRAGQuery, GraphRAGAnswer
        if self._rag is None:
            self._rag = GraphRAGQuery(self.km, llm=self.llm)
        return self._rag.query(text, mode=mode)

    def map_summary(self) -> str:
        return self.km.summary()

    # ── визуализация ─────────────────────────────────────────────────────

    def visualize(self, mode: str = "ascii", output: str = "infom_graph.html") -> str:
        """
        Визуализировать граф знаний.

        mode="ascii"  — цветной вывод в терминал
        mode="html"   — интерактивный HTML (D3.js), сохраняет файл
        """
        if mode == "html":
            from visualizer import render_html
            path = render_html(self.km, output)
            return f"HTML сохранён: {path}"
        else:
            from visualizer import render_full
            return render_full(self.km)

    # ── визуализация (текстовая) ──────────────────────────────────────────

    def _format_summary(self, result: QueryResult) -> str:
        lines = [
            f"Query: {result.query}",
            f"Questions generated: {len(result.questions.questions)}",
            f"Communities found:   {len(result.communities)}",
            f"HyperEdges found:    {len(result.hyper_edges)}",
        ]

        if result.communities:
            lines.append("\nTop communities:")
            for c in result.communities[:3]:
                shape = c.tangram.shape_class.value if c.tangram else "?"
                lines.append(
                    f"  [{shape:12s}] {c.label} "
                    f"(Q6={c.hex_id}, nodes={c.n_nodes})"
                )

        if result.hyper_edges:
            lines.append("\nTop hyper-edges:")
            for he in result.hyper_edges[:3]:
                s = he.shape.value if he.shape else "?"
                lines.append(f"  [{s:12s}] {he.label} ({he.n_nodes} nodes)")

        return "\n".join(lines)

    def print_geometry_legend(self) -> None:
        """Распечатать легенду геометрических форм."""
        legend = """
╔══════════════════════════════════════════════════════════════╗
║           InfoM — Геометрическая иерархия форм               ║
╠══════════╦══════╦═════════════════╦════════════════════════╣
║  Форма   ║ Нод  ║ Тип             ║ Источник               ║
╠══════════╬══════╬═════════════════╬════════════════════════╣
║ △  триан.║  3   ║ базовая триада  ║ meta2 / Tangram        ║
║ □  прямо.║  4   ║ уст. паттерн   ║ meta2 / Tangram        ║
║ ⬠  пятиу.║  5   ║ сложн. кластер ║ meta2 / Tangram        ║
║ ⬡  шести.║  6   ║ Q6-сота        ║ meta  / Q6 hexcore     ║
║ ☆  7-луч.║  7   ║ 3D звезда магов║ infom / Heptagram      ║
║ ✳  8-луч.║  8   ║ 3D роза ветров ║ infom / Octagram       ║
║ ∿  фракт.║  N   ║ граница группы ║ meta2 / Fractal (IFS)  ║
╚══════════╩══════╩═════════════════╩════════════════════════╝
        """
        print(legend)
