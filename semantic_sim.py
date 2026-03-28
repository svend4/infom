"""
SemanticSimulator — симуляция реального LLM+embedding адаптера.

Вместо случайных хэшей (MockLLMAdapter) здесь:
  1. SemanticEmbedder  — 32D векторы с реальной семантической геометрией.
     Оси 0-5 (ядро): [вычислительность, теоретичность, динамичность,
                       системность, дискретность, детерминизм]
     Оси 6-11 (ML/AI): [нейросетевость, seq2seq, эмбеддинг, обучение, поиск, генерация]
     Оси 12-17 (биол): [клеточность, ДНК/генетика, экосистемность, эволюционность,
                         метаболизм, нейробиология]
     Оси 18-23 (инфра): [транспорт, сеть, распред, поток, маршрут, физ.объект]
     Оси 24-29 (math):  [алгебра, анализ, геометрия, статистика, топология, граф-теория]
     Оси 30-31 (мета):  [сложность, временность]
     Похожие понятия → близкие векторы → правильная кластеризация Q6.
     Q6 проекция: average-pool 32→6 (не первые 6 элементов, а все равноценно).

  2. SemanticLLM       — синтез ответов: читает контекст графа и
     строит связный ответ из реальных сущностей и связей.

  3. SemanticAdapter   — полный LLMAdapter: embed + complete.

Использование:
    from semantic_sim import SemanticAdapter
    from graphrag_query import GraphRAGQuery
    rag = GraphRAGQuery(km, llm=SemanticAdapter())
"""
from __future__ import annotations
import math
import re
from llm_adapter import LLMAdapter, LLMResponse


# ── 1. Семантические оси (32D) ────────────────────────────────────────────────
#
# ЯДРО (0-5):   comp  theory  dynamic  system  discrete  determin
# ML/AI  (6-11): neural  seq2seq  embedding  learning  retrieval  generation
# БИОЛ  (12-17): cell  dna  ecosystem  evolution  metabolism  neurobio
# ИНФРА (18-23): transport  network  distributed  flow  routing  physical
# MATH  (24-29): algebra  calculus  geometry  statistics  topology  graph_theory
# МЕТА  (30-31): complexity  temporal
#
# Нормировка: каждый вектор единичной длины при хранении

def _v32(
    comp=0.0, theory=0.0, dynamic=0.0, system=0.0, discrete=0.0, determin=0.0,
    neural=0.0, seq2seq=0.0, embedding=0.0, learning=0.0, retrieval=0.0, generation=0.0,
    cell=0.0, dna=0.0, ecosystem=0.0, evolution=0.0, metabolism=0.0, neurobio=0.0,
    transport=0.0, network=0.0, distributed=0.0, flow=0.0, routing=0.0, physical=0.0,
    algebra=0.0, calculus=0.0, geometry=0.0, statistics=0.0, topology=0.0, graph_theory=0.0,
    complexity=0.0, temporal=0.0,
) -> list[float]:
    return [comp, theory, dynamic, system, discrete, determin,
            neural, seq2seq, embedding, learning, retrieval, generation,
            cell, dna, ecosystem, evolution, metabolism, neurobio,
            transport, network, distributed, flow, routing, physical,
            algebra, calculus, geometry, statistics, topology, graph_theory,
            complexity, temporal]


_SEMANTIC_VOCAB: dict[str, list[float]] = {
    # ── Математика / Физика ──────────────────────────────────────────────────
    "математика":        _v32(comp=0.3, theory=0.9, discrete=0.4, determin=0.9,
                              algebra=0.7, calculus=0.5, geometry=0.4, statistics=0.3, topology=0.3),
    "физика":            _v32(comp=0.2, theory=0.8, dynamic=0.4, system=0.3, determin=0.7,
                              calculus=0.5, algebra=0.3),
    "термодинамика":     _v32(comp=0.1, theory=0.7, dynamic=0.8, system=0.4, determin=0.5,
                              calculus=0.4, flow=0.5, metabolism=0.3),
    "механика":          _v32(comp=0.2, theory=0.7, dynamic=0.6, determin=0.8,
                              calculus=0.5, geometry=0.3),
    "статистика":        _v32(comp=0.4, theory=0.7, system=0.4, discrete=0.3, determin=0.6,
                              statistics=0.9, algebra=0.3),
    "линейная алгебра":  _v32(comp=0.5, theory=0.8, discrete=0.5, determin=0.9,
                              algebra=0.95, geometry=0.4, embedding=0.4),
    "теорвер":           _v32(comp=0.3, theory=0.8, dynamic=0.2, determin=0.3,
                              statistics=0.8, calculus=0.4),
    "оптимизация":       _v32(comp=0.7, theory=0.7, dynamic=0.6, discrete=0.5, determin=0.6,
                              calculus=0.6, learning=0.4, algebra=0.4),
    "градиент":          _v32(comp=0.6, theory=0.7, dynamic=0.7, discrete=0.4, determin=0.8,
                              calculus=0.8, learning=0.6, algebra=0.4),
    "матрица":           _v32(comp=0.5, theory=0.8, discrete=0.6, determin=0.9,
                              algebra=0.9, embedding=0.3),

    # ── Машинное обучение / AI ──────────────────────────────────────────────
    "нейросеть":    _v32(comp=0.9, theory=0.5, dynamic=0.8, system=0.7, discrete=0.6,
                         neural=0.95, learning=0.8, embedding=0.4, complexity=0.6),
    "трансформер":  _v32(comp=0.9, theory=0.6, dynamic=0.7, system=0.8, discrete=0.7,
                         neural=0.9, seq2seq=0.95, embedding=0.6, generation=0.7, complexity=0.7),
    "внимание":     _v32(comp=0.8, theory=0.6, dynamic=0.8, system=0.7, discrete=0.6,
                         neural=0.85, seq2seq=0.9, retrieval=0.5),
    "bert":         _v32(comp=0.9, theory=0.5, dynamic=0.6, system=0.7, discrete=0.7,
                         neural=0.9, seq2seq=0.8, embedding=0.8, determin=0.4),
    "gpt":          _v32(comp=0.9, theory=0.5, dynamic=0.7, system=0.7, discrete=0.7,
                         neural=0.9, seq2seq=0.85, generation=0.95, complexity=0.7),
    "llm":          _v32(comp=0.9, theory=0.5, dynamic=0.7, system=0.8, discrete=0.7,
                         neural=0.9, seq2seq=0.8, generation=0.9, embedding=0.5, complexity=0.8),
    "эмбеддинг":    _v32(comp=0.8, theory=0.6, system=0.5, discrete=0.7,
                         neural=0.6, embedding=0.95, retrieval=0.6, algebra=0.4, geometry=0.3),
    "обратное распространение": _v32(comp=0.8, theory=0.6, dynamic=0.9, discrete=0.6,
                         neural=0.8, learning=0.95, calculus=0.7, complexity=0.5),
    "функция потерь": _v32(comp=0.7, theory=0.7, dynamic=0.5, discrete=0.5,
                         neural=0.7, learning=0.8, calculus=0.6, statistics=0.4),
    "дообучение":   _v32(comp=0.8, theory=0.4, dynamic=0.8, system=0.5, discrete=0.6,
                         neural=0.8, learning=0.9, temporal=0.5),
    "обучение":     _v32(comp=0.7, theory=0.5, dynamic=0.9, system=0.5, discrete=0.5,
                         neural=0.7, learning=0.95, temporal=0.6),
    "классификация": _v32(comp=0.8, theory=0.5, dynamic=0.5, discrete=0.7,
                         neural=0.6, learning=0.7, statistics=0.5),
    "регрессия":    _v32(comp=0.7, theory=0.6, dynamic=0.4, discrete=0.5,
                         learning=0.7, statistics=0.6, calculus=0.4),
    "кластеризация": _v32(comp=0.8, theory=0.5, dynamic=0.5, system=0.7, discrete=0.6,
                         learning=0.6, graph_theory=0.4, statistics=0.4),

    # ── Алгоритмы / CS ──────────────────────────────────────────────────────
    "алгоритм":     _v32(comp=0.8, theory=0.6, dynamic=0.7, discrete=0.8, determin=0.9,
                         complexity=0.7, topology=0.2),
    "компилятор":   _v32(comp=0.9, theory=0.5, dynamic=0.7, discrete=0.9, determin=0.9,
                         complexity=0.5),
    "граф":         _v32(comp=0.7, theory=0.7, system=0.8, discrete=0.7, determin=0.8,
                         graph_theory=0.95, topology=0.5),
    "дерево":       _v32(comp=0.7, theory=0.6, system=0.7, discrete=0.8, determin=0.9,
                         graph_theory=0.85, algebra=0.2),
    "хеш":          _v32(comp=0.9, theory=0.4, discrete=0.9, determin=0.9,
                         retrieval=0.5),
    "поиск":        _v32(comp=0.9, theory=0.5, dynamic=0.7, discrete=0.8, determin=0.8,
                         retrieval=0.9, complexity=0.5),
    "сортировка":   _v32(comp=0.9, theory=0.5, dynamic=0.7, discrete=0.8, determin=0.9,
                         complexity=0.5),
    "индекс":       _v32(comp=0.8, theory=0.5, system=0.5, discrete=0.8, determin=0.8,
                         retrieval=0.7),
    "база данных":  _v32(comp=0.8, theory=0.4, dynamic=0.3, system=0.6, discrete=0.8,
                         retrieval=0.7, distributed=0.4),
    "векторная бд": _v32(comp=0.9, theory=0.5, system=0.6, discrete=0.8, determin=0.7,
                         embedding=0.8, retrieval=0.9),
    "hnsw":         _v32(comp=0.9, theory=0.6, system=0.6, discrete=0.8, determin=0.8,
                         retrieval=0.95, graph_theory=0.6, embedding=0.5, complexity=0.6),
    "lsh":          _v32(comp=0.9, theory=0.6, system=0.5, discrete=0.8, determin=0.6,
                         retrieval=0.9, statistics=0.4, embedding=0.5),
    "косинусное расстояние": _v32(comp=0.8, theory=0.6, discrete=0.7, determin=0.9,
                         embedding=0.7, geometry=0.6, algebra=0.4),

    # ── Граф знаний / RAG ───────────────────────────────────────────────────
    "граф знаний":  _v32(comp=0.8, theory=0.6, system=0.9, discrete=0.7, determin=0.7,
                         retrieval=0.7, graph_theory=0.8, embedding=0.4),
    "rag":          _v32(comp=0.9, theory=0.5, dynamic=0.7, system=0.7, discrete=0.7,
                         retrieval=0.9, generation=0.7, embedding=0.6),
    "graphrag":     _v32(comp=0.9, theory=0.6, dynamic=0.6, system=0.9, discrete=0.7,
                         retrieval=0.9, graph_theory=0.8, generation=0.6, embedding=0.5),
    "сообщество":   _v32(comp=0.6, theory=0.4, system=0.9, discrete=0.5,
                         graph_theory=0.7, topology=0.4),
    "кластер":      _v32(comp=0.7, theory=0.5, system=0.8, discrete=0.6,
                         graph_theory=0.6, statistics=0.4),
    "модульность":  _v32(comp=0.7, theory=0.7, system=0.8, discrete=0.6, determin=0.7,
                         graph_theory=0.8, statistics=0.5),
    "семантический поиск": _v32(comp=0.9, theory=0.5, dynamic=0.5, system=0.6, discrete=0.7,
                         retrieval=0.95, embedding=0.8, neural=0.5),

    # ── Биология / Экология ─────────────────────────────────────────────────
    "клетка":       _v32(comp=-0.5, theory=0.5, dynamic=0.7, system=0.6,
                         cell=0.95, metabolism=0.6, evolution=0.3),
    "днк":          _v32(comp=-0.3, theory=0.7, dynamic=0.4, system=0.5, determin=0.7,
                         dna=0.95, cell=0.5, evolution=0.5),
    "ген":          _v32(comp=-0.2, theory=0.7, dynamic=0.5, system=0.4, determin=0.6,
                         dna=0.85, cell=0.4, evolution=0.6),
    "белок":        _v32(comp=-0.4, theory=0.6, dynamic=0.6, system=0.4,
                         cell=0.7, metabolism=0.7, dna=0.4),
    "экосистема":   _v32(comp=-0.6, theory=0.4, dynamic=0.7, system=0.95,
                         ecosystem=0.95, evolution=0.6, flow=0.4),
    "эволюция":     _v32(comp=-0.5, theory=0.5, dynamic=0.9, system=0.8, determin=0.1,
                         evolution=0.95, ecosystem=0.5, temporal=0.8),
    "метаболизм":   _v32(comp=-0.5, theory=0.5, dynamic=0.8, system=0.5,
                         cell=0.6, metabolism=0.95, flow=0.5),
    "нейрон":       _v32(comp=-0.1, theory=0.6, dynamic=0.8, system=0.5,
                         neurobio=0.95, cell=0.5, neural=0.4),

    # ── Транспорт / Инфраструктура ──────────────────────────────────────────
    "транспорт":    _v32(comp=0.3, theory=0.1, dynamic=0.8, system=0.8, determin=0.6,
                         transport=0.95, network=0.6, routing=0.7, flow=0.6),
    "метро":        _v32(comp=0.4, theory=0.1, dynamic=0.7, system=0.8, discrete=0.4, determin=0.7,
                         transport=0.9, network=0.7, routing=0.6, physical=0.7),
    "инфраструктура": _v32(comp=0.4, theory=0.2, system=0.9, discrete=0.4, determin=0.6,
                         network=0.8, distributed=0.6, physical=0.7),
    "сеть":         _v32(comp=0.6, theory=0.3, dynamic=0.5, system=0.9, discrete=0.5,
                         network=0.95, distributed=0.7, graph_theory=0.5),
    "маршрут":      _v32(comp=0.4, theory=0.2, dynamic=0.7, system=0.6, determin=0.7,
                         routing=0.9, transport=0.6, graph_theory=0.4),

    # ── Общие / Нейтральные ─────────────────────────────────────────────────
    "данные":       _v32(comp=0.6, theory=0.4, system=0.5, discrete=0.7, determin=0.6,
                         retrieval=0.4, statistics=0.3),
    "модель":       _v32(comp=0.7, theory=0.6, dynamic=0.4, system=0.5, discrete=0.6,
                         neural=0.3, statistics=0.3, algebra=0.3),
    "система":      _v32(comp=0.5, theory=0.4, dynamic=0.5, system=0.8, discrete=0.5,
                         distributed=0.4, complexity=0.4),
    "архитектура":  _v32(comp=0.7, theory=0.5, system=0.7, discrete=0.6, determin=0.7,
                         complexity=0.4),
    "структура":    _v32(comp=0.5, theory=0.6, system=0.6, discrete=0.6, determin=0.8,
                         graph_theory=0.3, algebra=0.3),
    "процесс":      _v32(comp=0.4, theory=0.3, dynamic=0.9, system=0.5, discrete=0.4,
                         flow=0.5, temporal=0.6),
    "информация":   _v32(comp=0.5, theory=0.5, dynamic=0.4, system=0.6, discrete=0.6,
                         retrieval=0.4, statistics=0.3),
    "знание":       _v32(comp=0.4, theory=0.8, system=0.7, discrete=0.4,
                         retrieval=0.4, graph_theory=0.3),
    "язык":         _v32(comp=0.3, theory=0.7, dynamic=0.5, system=0.7,
                         seq2seq=0.5, generation=0.4, embedding=0.3),
    "текст":        _v32(comp=0.4, theory=0.5, system=0.5, discrete=0.5,
                         seq2seq=0.4, embedding=0.4, retrieval=0.3),
    "контекст":     _v32(comp=0.5, theory=0.5, dynamic=0.4, system=0.6,
                         seq2seq=0.5, retrieval=0.4),
    "токен":        _v32(comp=0.8, theory=0.4, discrete=0.9, determin=0.8,
                         seq2seq=0.7, embedding=0.5),
    "вектор":       _v32(comp=0.7, theory=0.7, discrete=0.7, determin=0.9,
                         embedding=0.7, algebra=0.6, geometry=0.4),
    "пространство": _v32(comp=0.4, theory=0.8, system=0.5, discrete=0.4, determin=0.7,
                         geometry=0.7, topology=0.6, algebra=0.4),
}


def _normalize(v: list[float]) -> list[float]:
    """Нормирует вектор до единичной длины."""
    n = math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x*y for x, y in zip(a, b))


def _cosine(a: list[float], b: list[float]) -> float:
    return max(-1.0, min(1.0, _dot(_normalize(a), _normalize(b))))


# Предвычислим нормированный словарь
_VOCAB_NORM: dict[str, list[float]] = {
    k: _normalize(v) for k, v in _SEMANTIC_VOCAB.items()
}


def _text_to_embedding(text: str) -> list[float]:
    """
    Преобразует текст в 32D семантический вектор.

    Алгоритм:
    1. Токенизируем текст (нижний регистр, разбиваем по пробелам/знакам)
    2. Для каждого токена ищем совпадение в _VOCAB_NORM (точное или частичное)
    3. Усредняем найденные векторы (взвешено по TF — чаще = важнее)
    4. Если ничего не найдено — детерминированный fallback через LCG
    """
    text_low = text.lower()
    # Убираем пунктуацию, разбиваем
    tokens = re.findall(r'[а-яёa-z0-9]+', text_low)

    matched: list[list[float]] = []
    weights: list[float] = []

    # Пробуем двух-трёхсловные фразы (bigrams)
    for i in range(len(tokens)):
        for size in [3, 2, 1]:
            phrase = " ".join(tokens[i:i+size])
            if phrase in _VOCAB_NORM:
                matched.append(_VOCAB_NORM[phrase])
                weights.append(float(size))  # длиннее фраза = больше вес
                break
        # Также частичное совпадение: содержится ли токен как подстрока ключа
        else:
            for key, vec in _VOCAB_NORM.items():
                if tokens[i] in key or key in tokens[i]:
                    matched.append(vec)
                    weights.append(0.6)
                    break

    if not matched:
        # LCG-fallback: детерминированный, но без семантики
        h = sum(ord(c) for c in text_low) * 2654435761
        result = []
        for i in range(6):
            h = (h * 1664525 + 1013904223) & 0xFFFFFFFF
            result.append((h / 0x7FFFFFFF) - 1.0)
        return _normalize(result)

    # Взвешенное среднее — размерность берём из первого найденного вектора
    total_w = sum(weights)
    ndim = len(matched[0])
    avg = [0.0] * ndim
    for vec, w in zip(matched, weights):
        for j in range(min(ndim, len(vec))):
            avg[j] += vec[j] * w / total_w

    return _normalize(avg)


# ── 2. Семантический LLM ─────────────────────────────────────────────────────

class SemanticLLM:
    """
    Синтез ответов с пониманием контекста.
    Читает реальные сущности, связи и архетипы из промпта
    и генерирует связный ответ — без рандома.
    """

    def complete(self, prompt: str) -> str:
        # Определяем тип запроса
        if "Сводка по сообществам" in prompt or re.search(r'\[w+/w+\]', prompt):
            return self._global_answer(prompt)
        return self._local_answer(prompt)

    # ── local ─────────────────────────────────────────────────────────────────

    def _local_answer(self, prompt: str) -> str:
        # Извлекаем вопрос
        q_match = re.search(r'Вопрос:\s*(.+)', prompt)
        question = q_match.group(1).strip() if q_match else ""

        # Извлекаем сущности
        ent_match = re.search(r'Сущности:\s*(.+)', prompt)
        entities_raw = ent_match.group(1).strip() if ent_match else ""
        entities = [e.strip() for e in entities_raw.split(",") if e.strip()][:5]

        # Извлекаем связи (до первой точки с запятой, чтобы не обрезать)
        rel_match = re.search(r'Связи:\s*(.+?)(?:\n|$)', prompt)
        relations_raw = rel_match.group(1).strip() if rel_match else ""
        relations = [r.strip() for r in relations_raw.split(";") if r.strip()][:4]

        # Архетипы
        arch_match = re.search(r'Архетипы:\s*(.+)', prompt)
        archetypes = arch_match.group(1).strip() if arch_match else ""

        # Форма / скелет
        shape_match = re.search(r'Форма кластера:\s*(\S+)', prompt)
        shape = shape_match.group(1) if shape_match else "?"
        skel_match = re.search(r'Скелет:\s*(\S+)', prompt)
        skeleton = skel_match.group(1) if skel_match else "?"

        # Q6
        q6_match = re.search(r'Q6-позиция:\s*(\d+)\s*\[([01]+)\]', prompt)
        q6 = q6_match.group(1) if q6_match else "?"

        # Ключевые вопросы QueryExpander
        exp_match = re.search(r'Уточняющие вопросы.*?:\s*([\s\S]+?)(?:\n\nДай|$)', prompt)
        expansion = exp_match.group(1).strip() if exp_match else ""

        # ── Синтез ответа ────────────────────────────────────────────────────
        parts: list[str] = []

        # 1. Главный тезис
        main_entity = entities[0] if entities else "объект"
        if entities:
            if len(entities) >= 3:
                cluster_str = ", ".join(entities[:-1]) + " и " + entities[-1]
                parts.append(
                    f"{main_entity} входит в кластер понятий: {cluster_str}."
                )
            else:
                parts.append(f"{main_entity} — ключевая сущность в данном домене.")

        # 2. Связи
        if relations:
            parts.append("Основные связи: " + "; ".join(relations[:3]) + ".")

        # 3. Геометрический профиль (осмысленно)
        shape_meanings = {
            "triangle":  "базовая триада из трёх взаимосвязанных понятий",
            "rectangle": "устойчивый четырёхэлементный паттерн",
            "pentagon":  "сложный пятиугольный кластер с перекрёстными связями",
            "hexagon":   "Q6-ячейка с шестью гранями семантического пространства",
            "polygon":   "многоугольный кластер (менее 3 нод или нестандартная форма)",
            "unknown":   "геометрия не определена",
        }
        shape_meaning = shape_meanings.get(shape, shape)
        if shape != "?":
            parts.append(
                f"Геометрически кластер представляет {shape_meaning} "
                f"(скелет: {skeleton}, Q6={q6})."
            )

        # 4. Архетипная характеристика
        arch_descriptions = {
            "ADEO": "алгоритмический / исследовательский",
            "ADCO": "алгоритмический / конвергентный",
            "ASCO": "структурный / конвергентный",
            "ASEO": "структурный / исследовательский",
            "MSCF": "морфический / потоковый",
            "MDEF": "живые системы / событийный",
            "MDCO": "морфический / конвергентный",
            "ADEF": "алгоритмический / событийный",
        }
        arch_codes = [a.strip() for a in archetypes.split(",") if a.strip()]
        if arch_codes:
            descs = [arch_descriptions.get(a, a) for a in arch_codes[:2]]
            parts.append(f"Доминирующий архетип: {', '.join(arch_codes)} — {'; '.join(descs)}.")

        # 5. Вывод на основе вопроса
        topic_words = re.findall(r'[а-яёА-ЯЁa-zA-Z]{4,}', question)
        if topic_words and entities:
            topic = topic_words[-1].lower()
            # Найти сущность, похожую на тему вопроса
            best = None
            for e in entities:
                if topic in e.lower() or e.lower() in topic:
                    best = e
                    break
            if best:
                parts.append(
                    f"Таким образом, {best} — это {shape_meaning}, "
                    f"семантически связанное с {', '.join(e for e in entities if e != best)[:60]}."
                )

        return " ".join(parts) if parts else f"Контекст содержит: {entities_raw}."

    # ── global ────────────────────────────────────────────────────────────────

    def _global_answer(self, prompt: str) -> str:
        # Парсим строки вида: [shape/arch] Q6=N skel=S dom=D: ноды...
        lines = re.findall(
            r'\[(\w+)/(\w+)\]\s+Q6=(\d+)\s+skel=(\w+)\s+dom=(\w+):\s*(.+)',
            prompt
        )

        arch_domain = {
            "ADEO": "алгоритмы и нейросети",
            "ADCO": "программные системы",
            "ASCO": "точные науки",
            "ASEO": "структурный анализ",
            "MSCF": "экосистемы и потоки",
            "MDEF": "живые организмы",
            "MDCO": "морфические системы",
            "ADEF": "событийные алгоритмы",
            "MSCO": "морфические структуры",
        }

        if not lines:
            return "Граф содержит несколько доменов знаний. Подробные данные недоступны."

        # Строим описание каждого домена
        community_lines = []
        domains_seen = []
        for shape, arch, q6, skel, dom, nodes_raw in lines:
            nodes = [n.strip() for n in nodes_raw.split(",")][:4]
            nodes_str = ", ".join(nodes)
            domain_name = arch_domain.get(arch, dom)
            domains_seen.append(domain_name)
            community_lines.append(
                f"  • [{shape}/{arch}] {domain_name}: {nodes_str}"
            )

        parts = [
            f"Граф охватывает {len(lines)} домен(а) знаний:",
            "\n".join(community_lines),
            "",
            f"Основные домены: {'; '.join(domains_seen[:4])}.",
        ]

        # Синтетическое заключение
        if len(lines) > 1:
            parts.append(
                "Между доменами прослеживаются семантические мосты: "
                "алгоритмические методы применяются в биологических и физических системах; "
                "математические основы объединяют все кластеры."
            )

        return "\n".join(parts)


# ── 3. Полный адаптер ─────────────────────────────────────────────────────────

class SemanticAdapter(LLMAdapter):
    """
    Полноценный симулятор LLM+embedding с реальной семантикой.

    embed()    → 6D вектор с семантической геометрией
    complete() → связный ответ, синтезированный из контекста
    """

    def __init__(self):
        self._llm = SemanticLLM()
        self.model = "semantic-sim-v1"

    def embed(self, text: str) -> list[float]:
        return _text_to_embedding(text)

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        # JSON-режим (для structured extraction)
        if '"entities"' in prompt or "'entities'" in prompt or "JSON" in prompt:
            return self._json_complete(prompt)
        text = self._llm.complete(prompt)
        return LLMResponse(text=text, model=self.model, tokens=len(text)//4)

    def _json_complete(self, prompt: str) -> LLMResponse:
        """Структурированное извлечение сущностей из текста."""
        import json as _json

        # Ищем текст после маркера "Текст:\n" (формат EXTRACT_PROMPT из indexer.py)
        text_match = re.search(r'Текст:\s*\n([\s\S]+?)(?:\n\n|$)', prompt)
        if not text_match:
            text_match = re.search(r'Текст:\s*"(.+?)"', prompt, re.DOTALL)
        if not text_match:
            text_match = re.search(r'текст[е:]?\s*[:\-]?\s*(.{20,200})', prompt, re.IGNORECASE | re.DOTALL)
        raw_text = text_match.group(1).strip() if text_match else prompt[-400:]

        text_low = raw_text.lower()

        # Существительные с заглавной буквы (реальные термины)
        cap_words = re.findall(r'[А-ЯЁ][а-яё]{3,}', raw_text)
        # Совпадения со словарём семантических понятий
        vocab_hits = [k for k in _SEMANTIC_VOCAB if k in text_low and len(k) > 4]
        # Латинские термины (аббревиатуры типа BERT, GPT, HNSW, RAG)
        latin_terms = re.findall(r'\b[A-Z]{2,}\b', raw_text)

        # Объединяем, убираем дубликаты, отфильтровываем JSON-слова
        _JSON_STOPWORDS = {"JSON", "entities", "relations", "label", "source", "target",
                           "weight", "type", "concept", "other", "place", "event", "person",
                           "Верни", "Текст", "Сущности", "Связи", "Извлеки"}
        combined = cap_words[:8] + [h.capitalize() for h in vocab_hits[:6]] + latin_terms[:4]
        entities = list(dict.fromkeys(
            e for e in combined if e not in _JSON_STOPWORDS and len(e) > 3
        ))[:10]
        if not entities:
            entities = ["Объект", "Концепция"]

        # Связи: ищем паттерны "A глагол B" в тексте
        rel_pattern = re.findall(
            r'([А-ЯЁ][а-яё]{3,})\s+(?:является|использует|строит|применяет|содержит'
            r'|обеспечивает|позволяет|измеряет|основан|применяется|представляет|объединяет)',
            raw_text
        )
        # Строим пары: каждый субъект → следующий найденный субъект
        relations = []
        for i in range(len(rel_pattern) - 1):
            a, b = rel_pattern[i], rel_pattern[i+1]
            if a != b and a not in _JSON_STOPWORDS and b not in _JSON_STOPWORDS:
                relations.append({"source": a, "target": b, "label": "связан", "weight": 0.7})
        if not relations and len(entities) >= 2:
            relations = [{"source": entities[0], "target": entities[1],
                          "label": "связан", "weight": 0.7}]

        result = {
            "entities": [{"id": e.lower()[:12], "label": e, "type": "concept"} for e in entities],
            "relations": relations,
        }
        text = _json.dumps(result, ensure_ascii=False)
        return LLMResponse(text=text, model=self.model, tokens=len(text)//4)


# ── Утилита: сравнение эмбеддингов ───────────────────────────────────────────

def compare_embeddings(pairs: list[tuple[str, str]]) -> None:
    """Выводит таблицу косинусного сходства между парами слов."""
    print("\nКосинусное сходство (SemanticEmbedder):")
    print(f"  {'Пара':<40}  Сходство  Интерпретация")
    print("  " + "─" * 65)
    for a, b in pairs:
        va = _text_to_embedding(a)
        vb = _text_to_embedding(b)
        sim = _cosine(va, vb)
        if sim > 0.85:   interp = "очень близко  ██████████"
        elif sim > 0.65: interp = "близко        ███████░░░"
        elif sim > 0.40: interp = "умеренно      █████░░░░░"
        elif sim > 0.10: interp = "слабо         ███░░░░░░░"
        else:            interp = "далеко        █░░░░░░░░░"
        print(f"  {a:<18} ↔ {b:<18}  {sim:+.3f}   {interp}")
