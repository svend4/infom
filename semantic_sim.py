"""
SemanticSimulator — симуляция реального LLM+embedding адаптера.

Вместо случайных хэшей (MockLLMAdapter) здесь:
  1. SemanticEmbedder  — 6D векторы с реальной семантической геометрией.
     Оси: [вычислительность, абстрактность, динамичность,
           системность, дискретность, детерминированность]
     Похожие понятия → близкие векторы → правильная кластеризация.

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


# ── 1. Семантические оси ──────────────────────────────────────────────────────
#
#  dim 0: вычислительность  (-1 = гуманитарное/биол,  +1 = алгоритм/код)
#  dim 1: теоретичность     (-1 = прикладное,         +1 = чистая теория)
#  dim 2: динамичность      (-1 = статичная структура, +1 = процесс/поток)
#  dim 3: системность       (-1 = атомарная сущность,  +1 = экосистема/сеть)
#  dim 4: дискретность      (-1 = непрерывное/аналог,  +1 = дискретное/цифровое)
#  dim 5: детерминизм       (-1 = вероятностное/хаос,  +1 = детерминированное)
#
# Формат: [comp, theory, dynamic, system, discrete, determin]
# Нормировка: каждый вектор единичной длины при хранении

_SEMANTIC_VOCAB: dict[str, list[float]] = {
    # ── Математика / Физика ──────────────────────────────────────────────────
    "математика":        [ 0.3,  0.9,  0.1,  0.2,  0.4,  0.9],
    "физика":            [ 0.2,  0.8,  0.4,  0.3,  0.2,  0.7],
    "термодинамика":     [ 0.1,  0.7,  0.8,  0.3,  0.1,  0.6],
    "механика":          [ 0.2,  0.7,  0.6,  0.2,  0.2,  0.8],
    "статистика":        [ 0.4,  0.7,  0.3,  0.4,  0.3,  0.7],
    "линейная алгебра":  [ 0.5,  0.8,  0.1,  0.2,  0.5,  0.9],
    "теорвер":           [ 0.3,  0.8,  0.2,  0.3,  0.3,  0.3],
    "оптимизация":       [ 0.7,  0.7,  0.6,  0.3,  0.5,  0.6],
    "градиент":          [ 0.6,  0.7,  0.7,  0.2,  0.4,  0.8],
    "матрица":           [ 0.5,  0.8,  0.1,  0.3,  0.6,  0.9],

    # ── Машинное обучение ────────────────────────────────────────────────────
    "нейросеть":         [ 0.9,  0.5,  0.8,  0.7,  0.6,  0.3],
    "трансформер":       [ 0.9,  0.6,  0.7,  0.8,  0.7,  0.3],
    "внимание":          [ 0.8,  0.6,  0.8,  0.7,  0.6,  0.3],
    "bert":              [ 0.9,  0.5,  0.6,  0.7,  0.7,  0.4],
    "gpt":               [ 0.9,  0.5,  0.7,  0.7,  0.7,  0.3],
    "llm":               [ 0.9,  0.5,  0.7,  0.8,  0.7,  0.2],
    "эмбеддинг":         [ 0.8,  0.6,  0.2,  0.5,  0.7,  0.5],
    "обратное распространение": [0.8, 0.6, 0.9, 0.3, 0.6, 0.7],
    "функция потерь":    [ 0.7,  0.7,  0.5,  0.3,  0.5,  0.6],
    "дообучение":        [ 0.8,  0.4,  0.8,  0.5,  0.6,  0.5],
    "обучение":          [ 0.7,  0.5,  0.9,  0.5,  0.5,  0.4],
    "классификация":     [ 0.8,  0.5,  0.5,  0.4,  0.7,  0.6],
    "регрессия":         [ 0.7,  0.6,  0.4,  0.3,  0.5,  0.7],
    "кластеризация":     [ 0.8,  0.5,  0.5,  0.7,  0.6,  0.5],

    # ── Алгоритмы / CS ──────────────────────────────────────────────────────
    "алгоритм":          [ 0.8,  0.6,  0.7,  0.3,  0.8,  0.9],
    "компилятор":        [ 0.9,  0.5,  0.7,  0.4,  0.9,  0.9],
    "граф":              [ 0.7,  0.7,  0.3,  0.8,  0.7,  0.8],
    "дерево":            [ 0.7,  0.6,  0.2,  0.7,  0.8,  0.9],
    "хеш":               [ 0.9,  0.4,  0.2,  0.2,  0.9,  0.9],
    "поиск":             [ 0.9,  0.5,  0.7,  0.4,  0.8,  0.8],
    "сортировка":        [ 0.9,  0.5,  0.7,  0.2,  0.8,  0.9],
    "индекс":            [ 0.8,  0.5,  0.2,  0.5,  0.8,  0.8],
    "база данных":       [ 0.8,  0.4,  0.3,  0.6,  0.8,  0.8],
    "векторная бд":      [ 0.9,  0.5,  0.4,  0.6,  0.8,  0.7],
    "hnsw":              [ 0.9,  0.6,  0.5,  0.6,  0.8,  0.8],
    "lsh":               [ 0.9,  0.6,  0.4,  0.5,  0.8,  0.6],
    "косинусное расстояние": [0.8, 0.6, 0.2, 0.3, 0.7, 0.9],

    # ── Граф знаний / RAG ───────────────────────────────────────────────────
    "граф знаний":       [ 0.8,  0.6,  0.3,  0.9,  0.7,  0.7],
    "rag":               [ 0.9,  0.5,  0.7,  0.7,  0.7,  0.5],
    "graphrag":          [ 0.9,  0.6,  0.6,  0.9,  0.7,  0.5],
    "сообщество":        [ 0.6,  0.4,  0.3,  0.9,  0.5,  0.5],
    "кластер":           [ 0.7,  0.5,  0.3,  0.8,  0.6,  0.6],
    "модульность":       [ 0.7,  0.7,  0.2,  0.8,  0.6,  0.7],
    "семантический поиск": [0.9, 0.5,  0.5,  0.6,  0.7,  0.6],

    # ── Биология / Экология ─────────────────────────────────────────────────
    "клетка":            [-0.5,  0.5,  0.7,  0.6, -0.3,  0.4],
    "днк":               [-0.3,  0.7,  0.4,  0.5, -0.2,  0.8],
    "ген":               [-0.2,  0.7,  0.5,  0.4, -0.1,  0.7],
    "белок":             [-0.4,  0.6,  0.6,  0.4, -0.3,  0.5],
    "экосистема":        [-0.6,  0.4,  0.7,  0.9, -0.5,  0.2],
    "эволюция":          [-0.5,  0.5,  0.9,  0.8, -0.4,  0.1],
    "метаболизм":        [-0.5,  0.5,  0.8,  0.5, -0.4,  0.4],
    "нейрон":            [-0.1,  0.6,  0.8,  0.5, -0.2,  0.3],

    # ── Транспорт / Инфраструктура ──────────────────────────────────────────
    "транспорт":         [ 0.3,  0.1,  0.8,  0.8,  0.3,  0.6],
    "метро":             [ 0.4,  0.1,  0.7,  0.8,  0.4,  0.7],
    "инфраструктура":    [ 0.4,  0.2,  0.3,  0.9,  0.4,  0.6],
    "сеть":              [ 0.6,  0.3,  0.5,  0.9,  0.5,  0.5],
    "маршрут":           [ 0.4,  0.2,  0.7,  0.6,  0.4,  0.7],

    # ── Общие / Нейтральные ─────────────────────────────────────────────────
    "данные":            [ 0.6,  0.4,  0.3,  0.5,  0.7,  0.6],
    "модель":            [ 0.7,  0.6,  0.4,  0.5,  0.6,  0.5],
    "система":           [ 0.5,  0.4,  0.5,  0.8,  0.5,  0.6],
    "архитектура":       [ 0.7,  0.5,  0.2,  0.7,  0.6,  0.7],
    "структура":         [ 0.5,  0.6,  0.1,  0.6,  0.6,  0.8],
    "процесс":           [ 0.4,  0.3,  0.9,  0.5,  0.4,  0.5],
    "информация":        [ 0.5,  0.5,  0.4,  0.6,  0.6,  0.5],
    "знание":            [ 0.4,  0.8,  0.2,  0.7,  0.4,  0.6],
    "язык":              [ 0.3,  0.7,  0.5,  0.7,  0.4,  0.4],
    "текст":             [ 0.4,  0.5,  0.3,  0.5,  0.5,  0.5],
    "контекст":          [ 0.5,  0.5,  0.4,  0.6,  0.5,  0.4],
    "токен":             [ 0.8,  0.4,  0.2,  0.3,  0.9,  0.8],
    "вектор":            [ 0.7,  0.7,  0.1,  0.3,  0.7,  0.9],
    "пространство":      [ 0.4,  0.8,  0.2,  0.5,  0.4,  0.7],
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
    Преобразует текст в 6D семантический вектор.

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

    # Взвешенное среднее
    total_w = sum(weights)
    avg = [0.0] * 6
    for vec, w in zip(matched, weights):
        for j in range(6):
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
