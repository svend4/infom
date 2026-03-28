"""
InfoM — демонстрация системы.
Два сценария:
  1. demo_manual()   — ручные эмбеддинги (показывает геометрию)
  2. demo_document() — индексация текстового документа
"""
import math
from pipeline    import InfoMPipeline, IndexConfig
from indexer     import DocumentIndexer, chunk_text
from llm_adapter import MockLLMAdapter
from signatures  import ShapeClass


def demo_manual():
    """Демо 1: ручные эмбеддинги, проверяем геометрическую иерархию."""
    pipeline = InfoMPipeline(config=IndexConfig(n_communities=4))
    pipeline.print_geometry_legend()

    # Эмбеддинги в [-1, 1] — после исправления embed_to_q6 распределяются по Q6
    nodes_data = [
        # Технологии (высокое значение по осям 0,1 — material+dynamic)
        ("n1",  "Алгоритм",        [ 0.9,  0.8,  0.7, -0.5, -0.6,  0.8], "ADEO"),
        ("n2",  "Нейросеть",       [ 0.8,  0.9,  0.8,  0.6, -0.7,  0.7], "ADCO"),
        ("n3",  "Компилятор",      [-0.2,  0.7,  0.9,  0.8, -0.5,  0.9], "ADCO"),
        # Города (высокое material+dynamic+complex)
        ("n4",  "Метро",           [ 0.8,  0.7,  0.6, -0.4,  0.8, -0.3], "MDCO"),
        ("n5",  "Транспорт",       [ 0.7,  0.8, -0.3, -0.6,  0.9, -0.5], "MDEO"),
        ("n6",  "Инфраструктура",  [ 0.9, -0.2,  0.8,  0.7,  0.6, -0.4], "MSCO"),
        # Наука (abstract+static)
        ("n7",  "Физика",          [-0.8, -0.7,  0.8,  0.9, -0.3,  0.6], "ASCO"),
        ("n8",  "Математика",      [-0.9, -0.8, -0.2,  0.8, -0.4,  0.9], "ASEO"),
        ("n9",  "Термодинамика",   [-0.7, -0.6,  0.7,  0.8, -0.2, -0.3], "ASCO"),
        # Биология (material+dynamic)
        ("n10", "Клетка",          [ 0.7,  0.6, -0.5, -0.7, -0.8, -0.6], "MDEF"),
        ("n11", "ДНК",             [ 0.8, -0.3, -0.6,  0.6, -0.7, -0.5], "MSEO"),
        ("n12", "Экосистема",      [ 0.9, -0.4,  0.7, -0.5, -0.6, -0.7], "MSCF"),
    ]

    for nid, label, emb, arch in nodes_data:
        pipeline.add_node(nid, label, emb, arch)

    edges_data = [
        ("n1",  "n2",  "является основой",   0.9),
        ("n2",  "n3",  "компилируется через", 0.7),
        ("n1",  "n3",  "реализуется в",       0.8),
        ("n4",  "n5",  "использует",          0.9),
        ("n5",  "n6",  "часть",               0.8),
        ("n4",  "n6",  "включает",            0.7),
        ("n7",  "n8",  "формализует",         0.9),
        ("n8",  "n9",  "применяется в",       0.8),
        ("n7",  "n9",  "объясняет",           0.9),
        ("n10", "n11", "содержит",            0.9),
        ("n11", "n12", "формирует",           0.7),
        ("n10", "n12", "часть",               0.8),
        ("n1",  "n7",  "изучает",             0.5),
        ("n2",  "n10", "моделирует",          0.4),
        ("n6",  "n4",  "обслуживает",         0.6),
    ]

    for src, tgt, lbl, w in edges_data:
        pipeline.add_edge(src, tgt, lbl, w)

    print("\n" + "="*60)
    print("Строим граф знаний...")
    pipeline.build()
    print(pipeline.map_summary())

    # Q6 позиции нод
    print("\nQ6 позиции нод:")
    for node in pipeline.km.nodes.values():
        hid   = node.hex_id
        bits  = node.hex_sig.bits if node.hex_sig else ()
        print(f"  {node.label:20s} hex_id={hid:2d}  bits={bits}")

    # сообщества
    print("\nСообщества:")
    for comm in pipeline.km.communities.values():
        shape  = comm.tangram.shape_class.value if comm.tangram else "?"
        skel   = comm.octagram.skeleton_type.value if comm.octagram else "?"
        labels = [n.label for n in comm.nodes]
        print(f"  [{shape:12s}|{skel:8s}] Q6={comm.hex_id:2d}  {labels}")

    # гиперрёбра
    print(f"\nГиперрёбра ({len(pipeline.km.hyper_edges)}):")
    for he in pipeline.km.hyper_edges[:6]:
        shape  = he.shape.value if he.shape else "?"
        dom    = he.heptagram.dominant_ray.label if he.heptagram else "?"
        labels = [pipeline.km.nodes[n].label for n in he.nodes
                  if n in pipeline.km.nodes]
        print(f"  [{shape:12s}] dominant={dom:12s}  {labels}")

    # границы
    print(f"\nФрактальные границы ({len(pipeline.km.borders)}):")
    for b in pipeline.km.borders[:4]:
        fd  = b.fractal.fd_box
        tag = "размытая" if fd > 1.5 else "чёткая"
        print(f"  {b.community_a} ↔ {b.community_b}  fd_box={fd:.3f} ({tag})")

    # запрос
    print("\n" + "="*60)
    print("Запрос: 'алгоритм обучения'")
    print("="*60)
    result = pipeline.query("алгоритм обучения", node_id="n1")
    print(result.summary)

    # поиск по форме
    print("\n" + "="*60)
    print("Кластеры-треугольники (3 ноды):")
    tri = pipeline.search_by_shape(ShapeClass.TRIANGLE)
    print(f"  HyperEdges: {len(tri.hyper_edges)}")
    for he in tri.hyper_edges[:3]:
        labels = [pipeline.km.nodes[n].label for n in he.nodes
                  if n in pipeline.km.nodes]
        print(f"    {labels}")

    print("\nКластеры-прямоугольники (4 ноды):")
    rect = pipeline.search_by_shape(ShapeClass.RECTANGLE)
    print(f"  Communities: {len(rect.communities)},  HyperEdges: {len(rect.hyper_edges)}")


def demo_document():
    """Демо 2: индексация текстового документа через DocumentIndexer."""
    print("\n" + "="*60)
    print("DEMO 2: Индексация документа")
    print("="*60)

    text = """
    Нейронные сети являются основой современного машинного обучения.
    Алгоритм обратного распространения ошибки позволяет обучать глубокие сети.
    Трансформеры произвели революцию в обработке естественного языка.
    Архитектура трансформера использует механизм внимания (attention).
    Механизм внимания позволяет модели фокусироваться на релевантных частях входа.
    Большие языковые модели (LLM) обучаются на огромных текстовых корпусах.
    GPT и BERT являются примерами трансформерных архитектур.
    Векторные эмбеддинги представляют слова в многомерном пространстве.
    Семантический поиск использует эмбеддинги для нахождения похожих документов.
    Граф знаний структурирует информацию в виде сущностей и отношений.
    GraphRAG объединяет графы знаний с генеративным AI для улучшения поиска.
    """

    indexer = DocumentIndexer(
        llm          = MockLLMAdapter(),
        chunk_size   = 300,
        chunk_overlap = 50,
    )

    # показываем чанки
    chunks = chunk_text(text.strip(), chunk_size=300)
    print(f"\nЧанков: {len(chunks)}")
    for c in chunks:
        print(f"  [{c.id}] {c.text[:80]}...")

    print("\nИндексируем...")
    km, result = indexer.index(text.strip())

    print(f"\nРезультат индексации:")
    print(f"  Чанков:       {result.n_chunks}")
    print(f"  Сущностей:    {result.n_entities}")
    print(f"  Связей:       {result.n_relations}")
    print(f"  Нод в графе:  {result.n_nodes}")
    print(f"  Рёбер:        {result.n_edges}")
    print(f"  Сообществ:    {result.n_communities}")
    print()
    print(km.summary())


def demo_visualize():
    """Демо 3: визуализация — ASCII + HTML."""
    import os
    pipeline = InfoMPipeline(config=IndexConfig(n_communities=4))

    nodes_data = [
        ("n1",  "Алгоритм",        [ 0.9,  0.8,  0.7, -0.5, -0.6,  0.8], "ADEO"),
        ("n2",  "Нейросеть",       [ 0.8,  0.9,  0.8,  0.6, -0.7,  0.7], "ADCO"),
        ("n3",  "Компилятор",      [-0.2,  0.7,  0.9,  0.8, -0.5,  0.9], "ADCO"),
        ("n4",  "Метро",           [ 0.8,  0.7,  0.6, -0.4,  0.8, -0.3], "MDCO"),
        ("n5",  "Транспорт",       [ 0.7,  0.8, -0.3, -0.6,  0.9, -0.5], "MDEO"),
        ("n6",  "Инфраструктура",  [ 0.9, -0.2,  0.8,  0.7,  0.6, -0.4], "MSCO"),
        ("n7",  "Физика",          [-0.8, -0.7,  0.8,  0.9, -0.3,  0.6], "ASCO"),
        ("n8",  "Математика",      [-0.9, -0.8, -0.2,  0.8, -0.4,  0.9], "ASEO"),
        ("n9",  "Термодинамика",   [-0.7, -0.6,  0.7,  0.8, -0.2, -0.3], "ASCO"),
        ("n10", "Клетка",          [ 0.7,  0.6, -0.5, -0.7, -0.8, -0.6], "MDEF"),
        ("n11", "ДНК",             [ 0.8, -0.3, -0.6,  0.6, -0.7, -0.5], "MSEO"),
        ("n12", "Экосистема",      [ 0.9, -0.4,  0.7, -0.5, -0.6, -0.7], "MSCF"),
    ]
    for nid, label, emb, arch in nodes_data:
        pipeline.add_node(nid, label, emb, arch)

    for src, tgt, lbl, w in [
        ("n1","n2","основа",0.9), ("n2","n3","компилируется",0.7),
        ("n1","n3","реализуется",0.8), ("n4","n5","использует",0.9),
        ("n5","n6","часть",0.8), ("n4","n6","включает",0.7),
        ("n7","n8","формализует",0.9), ("n8","n9","применяется",0.8),
        ("n7","n9","объясняет",0.9), ("n10","n11","содержит",0.9),
        ("n11","n12","формирует",0.7), ("n10","n12","часть",0.8),
        ("n1","n7","изучает",0.5), ("n2","n10","моделирует",0.4),
        ("n6","n4","обслуживает",0.6),
    ]:
        pipeline.add_edge(src, tgt, lbl, w)

    pipeline.build()

    # ASCII визуализация
    print("\n" + "="*70)
    print("ASCII ВИЗУАЛИЗАЦИЯ")
    print("="*70)
    print(pipeline.visualize(mode="ascii"))

    # HTML визуализация
    html_path = os.path.join(os.path.dirname(__file__), "infom_graph.html")
    msg = pipeline.visualize(mode="html", output=html_path)
    print(f"\n{msg}")
    print("Откройте в браузере для интерактивного просмотра.")


if __name__ == "__main__":
    demo_visualize()
