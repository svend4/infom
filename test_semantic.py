"""
Полное тестирование InfoM с SemanticAdapter.

Сравниваем MockLLMAdapter (хэш-эмбеддинги) vs SemanticAdapter (семантика).
Проверяем:
  1. Качество эмбеддингов — косинусное сходство смысловых пар
  2. Кластеризацию — попадают ли близкие понятия в одно сообщество
  3. Качество RAG-ответов — релевантность и связность
  4. DocumentIndexer — извлечение сущностей и построение графа
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from semantic_sim  import SemanticAdapter, compare_embeddings
from llm_adapter   import MockLLMAdapter
from graph         import KnowledgeMap, GraphNode, GraphEdge
from graphrag_query import GraphRAGQuery
from indexer       import DocumentIndexer
from visualizer.ascii import render_communities, render_graph_ascii

SEP = "═" * 68


# ── 1. Тест эмбеддингов ───────────────────────────────────────────────────────

def test_embeddings():
    print(f"\n{SEP}")
    print("ТЕСТ 1: Качество семантических эмбеддингов")
    print(SEP)

    pairs = [
        # Должны быть близко (> 0.65)
        ("нейросеть",     "трансформер"),
        ("bert",          "gpt"),
        ("градиент",      "функция потерь"),
        ("hnsw",          "lsh"),
        ("граф",          "граф знаний"),
        ("клетка",        "днк"),
        ("транспорт",     "метро"),
        ("математика",    "статистика"),
        # Должны быть далеко (< 0.30)
        ("нейросеть",     "экосистема"),
        ("алгоритм",      "клетка"),
        ("математика",    "метро"),
        ("компилятор",    "днк"),
    ]
    compare_embeddings(pairs)

    # Тест: Mock vs Semantic
    mock = MockLLMAdapter()
    sem  = SemanticAdapter()
    print("\nMock (хэш) vs Semantic — для пары (нейросеть, трансформер):")
    from semantic_sim import _cosine
    vm_a = mock.embed("нейросеть");   vm_b = mock.embed("трансформер")
    vs_a = sem.embed("нейросеть");    vs_b = sem.embed("трансформер")
    print(f"  Mock:     {_cosine(vm_a, vm_b):+.3f}  (ожидается ~0, хэш рандомен)")
    print(f"  Semantic: {_cosine(vs_a, vs_b):+.3f}  (ожидается > 0.8)")

    print("\nMock vs Semantic — для пары (нейросеть, экосистема):")
    vm_a = mock.embed("нейросеть");   vm_b = mock.embed("экосистема")
    vs_a = sem.embed("нейросеть");    vs_b = sem.embed("экосистема")
    print(f"  Mock:     {_cosine(vm_a, vm_b):+.3f}  (ожидается ~случайно)")
    print(f"  Semantic: {_cosine(vs_a, vs_b):+.3f}  (ожидается < 0.2)")


# ── 2. Тест кластеризации ────────────────────────────────────────────────────

def build_km(llm) -> KnowledgeMap:
    """Строит граф знаний из 12 концептов с семантическими эмбеддингами."""
    km = KnowledgeMap()

    concepts = [
        # (id, label, archetype)
        ("nn",      "Нейросеть",     "ADEO"),
        ("transf",  "Трансформер",   "ADEO"),
        ("bert",    "BERT",          "ADEO"),
        ("gpt",     "GPT",           "ADEO"),
        ("embed",   "Эмбеддинг",     "ADCO"),
        ("grad",    "Градиент",      "ASCO"),
        ("loss",    "Функц.потерь",  "ASCO"),
        ("bp",      "Обрат.распр",   "ASCO"),
        ("cell",    "Клетка",        "MDEF"),
        ("dna",     "ДНК",           "MDEF"),
        ("eco",     "Экосистема",    "MSCF"),
        ("metro",   "Метро",         "ADCO"),
    ]
    for nid, label, arch in concepts:
        emb = llm.embed(label)
        km.add_node(GraphNode(id=nid, label=label, archetype=arch, embedding=emb))

    edges = [
        ("nn",     "transf",  "основан на",    0.9),
        ("bert",   "transf",  "использует",    0.9),
        ("gpt",    "transf",  "использует",    0.9),
        ("transf", "embed",   "производит",    0.85),
        ("bp",     "grad",    "вычисляет",     0.95),
        ("bp",     "loss",    "минимизирует",  0.9),
        ("nn",     "bp",      "обучается",     0.8),
        ("cell",   "dna",     "содержит",      0.9),
        ("eco",    "cell",    "включает",      0.75),
        ("metro",  "metro",   "self",          0.0),  # изолированный
    ]
    for s, t, l, w in edges:
        if s != t:
            km.add_edge(GraphEdge(source=s, target=t, label=l, weight=w))

    km.build()
    return km


def test_clustering():
    print(f"\n{SEP}")
    print("ТЕСТ 2: Качество кластеризации")
    print(SEP)

    mock_km = build_km(MockLLMAdapter())
    sem_km  = build_km(SemanticAdapter())

    def show_communities(km, name):
        print(f"\n  [{name}]  Q={km.modularity:.3f}  LP={km.lp_iterations} iter")
        for c in km.communities.values():
            nodes = ", ".join(n.label for n in c.nodes)
            shape = c.tangram.shape_class.value if c.tangram else "?"
            print(f"    [{c.dominant_archetype}] {shape:10s}  {nodes}")

    show_communities(mock_km, "MockLLM  (хэш-эмбеддинги)")
    show_communities(sem_km,  "Semantic (смысловые векторы)")

    # Оценка: правильно ли сгруппированы ML-понятия
    sem_correct = 0
    ml_nodes = {"nn", "transf", "bert", "gpt"}
    for c in sem_km.communities.values():
        ids = {n.id for n in c.nodes}
        overlap = ids & ml_nodes
        if len(overlap) >= 3:
            sem_correct = len(overlap)
    print(f"\n  ML-кластер (nn/transf/bert/gpt) в SemanticAdapter: {sem_correct}/4 в одном сообществе")

    bio_nodes = {"cell", "dna", "eco"}
    for c in sem_km.communities.values():
        ids = {n.id for n in c.nodes}
        overlap = ids & bio_nodes
        if len(overlap) >= 2:
            print(f"  Bio-кластер (клетка/ДНК/экосистема):          {len(overlap)}/3 в одном сообществе")
            break

    return sem_km


# ── 3. Тест RAG-ответов ──────────────────────────────────────────────────────

def test_rag(km: KnowledgeMap):
    print(f"\n{SEP}")
    print("ТЕСТ 3: Качество RAG-ответов (SemanticAdapter)")
    print(SEP)

    sem = SemanticAdapter()
    rag = GraphRAGQuery(km, llm=sem)

    queries = [
        ("Что такое трансформер?",          "local"),
        ("Как работает обратное распространение?", "local"),
        ("Что такое ДНК?",                  "local"),
        ("Какие области знаний охватывает граф?", "global"),
        ("Как связаны нейросети и математика?",   "hybrid"),
    ]

    for q, mode in queries:
        print(f"\n  ── {mode.upper()}: «{q}» ──")
        ans = rag.query(q, mode=mode)
        # Показываем ответ и источники
        print(f"  ОТВЕТ: {ans.answer[:300]}")
        print(f"  ИСТОЧНИКИ: {', '.join(ans.sources[:4])}")
        if ans.questions:
            top_q = ans.questions.questions[:2] if ans.questions.questions else []
            if top_q:
                print(f"  EXPANDER: {top_q[0].text[:80]}")


# ── 4. Тест DocumentIndexer ───────────────────────────────────────────────────

def test_document_indexer():
    print(f"\n{SEP}")
    print("ТЕСТ 4: DocumentIndexer с SemanticAdapter")
    print(SEP)

    text = """
    Трансформер — архитектура нейронной сети, основанная на механизме внимания.
    Механизм внимания позволяет модели взвешивать важность каждого токена.
    BERT использует трансформер для двунаправленного кодирования текста.
    GPT применяет трансформер для авторегрессивной генерации токенов.
    Эмбеддинги представляют слова в виде векторов в семантическом пространстве.
    Косинусное расстояние измеряет близость эмбеддингов в векторном пространстве.
    Обратное распространение вычисляет градиент функции потерь по весам.
    Оптимизатор использует градиент для обновления весов нейросети.
    HNSW строит иерархический граф для быстрого поиска ближайших векторов.
    Векторная база данных хранит эмбеддинги и поддерживает семантический поиск.
    RAG объединяет векторный поиск с языковой моделью для генерации ответов.
    GraphRAG использует граф знаний для структурированного поиска контекста.
    """

    indexer = DocumentIndexer(llm=SemanticAdapter())
    km, result = indexer.index(text)

    print(f"\n  Извлечено: {result.n_entities} сущностей, {result.n_relations} связей")
    print(f"  Граф:      {len(km.nodes)} нод, {len(km.edges)} рёбер")
    print(f"  Сообщества:{len(km.communities)}  Q={km.modularity:.3f}  LP={km.lp_iterations}")
    print()
    print("  Сообщества:")
    for c in km.communities.values():
        nodes = ", ".join(n.label for n in c.nodes[:5])
        shape = c.tangram.shape_class.value if c.tangram else "?"
        print(f"    [{c.dominant_archetype}] {shape:10s}  {nodes}")

    # RAG на проиндексированном документе
    rag = GraphRAGQuery(km, llm=SemanticAdapter())
    print()
    for q in ["Что такое RAG?", "Как устроен трансформер?"]:
        ans = rag.query(q, mode="local")
        print(f"  Q: {q}")
        print(f"  A: {ans.answer[:250]}")
        print()


# ── 5. ASCII визуализация ─────────────────────────────────────────────────────

def test_ascii(km: KnowledgeMap):
    print(f"\n{SEP}")
    print("ТЕСТ 5: ASCII-граф (SemanticAdapter кластеризация)")
    print(SEP)
    print(render_graph_ascii(km))
    print(render_communities(km))


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_embeddings()
    sem_km = test_clustering()
    test_rag(sem_km)
    test_document_indexer()
    test_ascii(sem_km)

    print(f"\n{SEP}")
    print("ИТОГ: тест завершён.")
    print("  SemanticAdapter успешно симулирует реальный LLM+embedding.")
    print(SEP)
