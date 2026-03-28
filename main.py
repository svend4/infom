"""
InfoM — демонстрация системы.
"""
import sys
import math
from pipeline import InfoMPipeline, IndexConfig
from signatures import ShapeClass


def demo_geometry():
    """Демо: геометрическая иерархия форм."""
    pipeline = InfoMPipeline(config=IndexConfig(n_communities=6))
    pipeline.print_geometry_legend()

    # Создаём тестовые ноды с эмбеддингами (симулируем 6D Q6)
    nodes_data = [
        # Технологии
        ("n1",  "Алгоритм",       [1, 1, 1, 0, 0, 1], "ADEO"),
        ("n2",  "Нейросеть",      [1, 1, 1, 1, 0, 1], "ADCO"),
        ("n3",  "Компилятор",     [0, 1, 1, 1, 0, 1], "ADCO"),
        # Города
        ("n4",  "Метро",          [1, 1, 1, 0, 1, 0], "MDCO"),
        ("n5",  "Транспорт",      [1, 1, 0, 0, 1, 0], "MDEO"),
        ("n6",  "Инфраструктура", [1, 0, 1, 1, 1, 0], "MSCO"),
        # Наука
        ("n7",  "Физика",         [0, 0, 1, 1, 0, 1], "ASCO"),
        ("n8",  "Математика",     [0, 0, 0, 1, 0, 1], "ASEO"),
        ("n9",  "Термодинамика",  [0, 0, 1, 1, 0, 0], "ASCO"),
        # Биология
        ("n10", "Клетка",         [1, 1, 0, 0, 0, 0], "MDEF"),
        ("n11", "ДНК",            [1, 0, 0, 1, 0, 0], "MSEO"),
        ("n12", "Экосистема",     [1, 0, 1, 0, 0, 0], "MSCF"),
    ]

    for nid, label, emb, arch in nodes_data:
        pipeline.add_node(nid, label, [float(x) for x in emb], arch)

    # Рёбра
    edges_data = [
        ("n1", "n2", "является основой", 0.9),
        ("n2", "n3", "компилируется через", 0.7),
        ("n1", "n3", "реализуется", 0.8),
        ("n4", "n5", "использует", 0.9),
        ("n5", "n6", "часть", 0.8),
        ("n4", "n6", "включает", 0.7),
        ("n7", "n8", "формализует", 0.9),
        ("n8", "n9", "применяется в", 0.8),
        ("n7", "n9", "объясняет", 0.9),
        ("n10","n11","содержит", 0.9),
        ("n11","n12","формирует", 0.7),
        ("n10","n12","часть", 0.8),
        # межтематические связи
        ("n1", "n7", "изучает", 0.5),
        ("n2", "n10","моделирует", 0.4),
    ]

    for src, tgt, lbl, w in edges_data:
        pipeline.add_edge(src, tgt, lbl, w)

    # Построить граф
    print("Строим граф знаний...")
    pipeline.build()
    print(pipeline.map_summary())

    # Запрос
    print("\n" + "="*60)
    print("Запрос: 'алгоритм обучения'")
    print("="*60)
    result = pipeline.query("алгоритм обучения", node_id="n1")
    print(result.summary)

    print("\n" + "="*60)
    print("Вопросы по 16 архетипам + 7 Heptagram-измерениям:")
    print("="*60)
    print(result.questions.to_markdown()[:1500])

    # Поиск по форме
    print("\n" + "="*60)
    print("Кластеры-треугольники (3 ноды):")
    print("="*60)
    triangles = pipeline.search_by_shape(ShapeClass.TRIANGLE)
    print(f"  Найдено: {triangles.total}")
    for he in triangles.hyper_edges[:3]:
        print(f"  HyperEdge: {he.label} — {he.nodes}")


if __name__ == "__main__":
    demo_geometry()
