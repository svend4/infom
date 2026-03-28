"""
QueryExpander — разворачивает простой запрос в структурированное дерево вопросов
по 16 архетипам + 7 геометрическим измерениям (HeptagramSignature).
Портировано и расширено из pseudorag/core/query_expander.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import re


from archetypes.archetypes import Archetype, ARCHETYPES, find_by_keyword

# ── шаблоны вопросов по 16 архетипам ────────────────────────────────────────

QUESTION_TEMPLATES: dict[str, list[str]] = {
    "MSEO": [
        "Какова базовая структура {topic}?",
        "Из каких элементарных компонентов состоит {topic}?",
        "Какова геометрия/форма {topic}?",
    ],
    "MSEF": [
        "Как распределены элементы {topic}?",
        "Какова степень хаотичности {topic}?",
        "Какие случайные вариации существуют в {topic}?",
    ],
    "MSCO": [
        "Какова архитектура {topic}?",
        "Как организована иерархия в {topic}?",
        "Какие подсистемы входят в {topic}?",
    ],
    "MSCF": [
        "Какова экосистема вокруг {topic}?",
        "Какие природные процессы связаны с {topic}?",
        "Как {topic} взаимодействует со средой?",
    ],
    "MDEO": [
        "Как работает базовый механизм {topic}?",
        "Какова причинно-следственная цепочка в {topic}?",
        "Что приводит {topic} в движение?",
    ],
    "MDEF": [
        "Как {topic} адаптируется к изменениям?",
        "Какие жизненные циклы характерны для {topic}?",
        "Как {topic} реагирует на внешние воздействия?",
    ],
    "MDCO": [
        "Как устроена техническая система {topic}?",
        "Какие технологии используются в {topic}?",
        "Как автоматизированы процессы {topic}?",
    ],
    "MDCF": [
        "Как развивается {topic} как система?",
        "Какова социальная динамика {topic}?",
        "Какие сети связей существуют в {topic}?",
    ],
    "ASEO": [
        "Какие фундаментальные принципы лежат в основе {topic}?",
        "Какие аксиомы или законы управляют {topic}?",
        "Что является неизменной истиной о {topic}?",
    ],
    "ASEF": [
        "Какие архетипические паттерны проявляются в {topic}?",
        "Какие символы или образы связаны с {topic}?",
        "Какой прообраз лежит в основе {topic}?",
    ],
    "ASCO": [
        "Какие теории объясняют {topic}?",
        "Какова научная модель {topic}?",
        "В каком концептуальном фреймворке существует {topic}?",
    ],
    "ASCF": [
        "Какова культурная роль {topic}?",
        "Как {topic} воспринимается в разных культурах?",
        "Какие ценности связаны с {topic}?",
    ],
    "ADEO": [
        "Каков алгоритм работы с {topic}?",
        "Какие шаги необходимы для {topic}?",
        "Как пошагово описать процесс {topic}?",
    ],
    "ADEF": [
        "Какие интуитивные/эмоциональные аспекты есть у {topic}?",
        "Как {topic} ощущается субъективно?",
        "Какие неформальные знания существуют о {topic}?",
    ],
    "ADCO": [
        "Как {topic} реализуется программно?",
        "Какова архитектура системы {topic}?",
        "Какие алгоритмические паттерны используются в {topic}?",
    ],
    "ADCF": [
        "Какова социальная роль {topic}?",
        "Как {topic} влияет на общество?",
        "Какие общественные движения связаны с {topic}?",
    ],
}

# ── 7 дополнительных вопросов (Heptagram — 7 измерений) ─────────────────────

HEPTAGRAM_QUESTIONS: dict[str, str] = {
    "strength":   "Насколько сильна связь {topic} с данным контекстом?",
    "direction":  "Является ли влияние {topic} односторонним или взаимным?",
    "temporal":   "Как изменяется {topic} во времени?",
    "confidence": "Насколько достоверна информация о {topic}?",
    "scale":      "На каком масштабе (микро/макро) работает {topic}?",
    "context":    "В каком контексте наиболее релевантен {topic}?",
    "source":     "Каков источник знания о {topic}?",
}


@dataclass
class Question:
    id:           str
    text:         str
    archetype:    str
    priority:     int
    answer_type:  str    # "text", "list", "number", "bool"
    dimension:    str = ""   # для Heptagram-вопросов


@dataclass
class QuestionTree:
    topic:     str
    questions: list[Question] = field(default_factory=list)

    def by_archetype(self, code: str) -> list[Question]:
        return [q for q in self.questions if q.archetype == code]

    def by_priority(self, min_p: int = 3) -> list[Question]:
        return [q for q in self.questions if q.priority >= min_p]

    def to_json(self) -> str:
        return json.dumps({
            "topic": self.topic,
            "total": len(self.questions),
            "questions": [
                {"id": q.id, "text": q.text,
                 "archetype": q.archetype, "priority": q.priority}
                for q in self.questions
            ],
        }, ensure_ascii=False, indent=2)

    def to_markdown(self) -> str:
        lines = [f"# Вопросы по теме: {self.topic}\n"]
        current = ""
        for q in sorted(self.questions, key=lambda x: (x.archetype, x.id)):
            if q.archetype != current:
                current = q.archetype
                lines.append(f"\n## Архетип {current}\n")
            lines.append(f"- [{q.priority}★] {q.text}")
        return "\n".join(lines)


class QueryExpander:
    """
    Преобразует простой запрос в структурированное дерево вопросов.
    Использует 16 архетипов + 7 измерений Heptagram.
    """

    MIN_RELEVANCE = 0.2

    DOMAIN_KEYWORDS: dict[str, list[str]] = {
        "urbanism":   ["город", "city", "урбан", "транспорт", "метро", "улица"],
        "biology":    ["организм", "клетка", "ген", "животн", "растен", "биолог"],
        "technology": ["алгоритм", "программ", "код", "систем", "технолог", "ИИ", "AI"],
        "geography":  ["страна", "регион", "климат", "геогр", "территор", "карт"],
        "culture":    ["культур", "искусств", "язык", "религ", "традиц", "история"],
        "science":    ["теория", "закон", "модель", "эксперим", "наук", "физик", "химик"],
        "society":    ["общество", "социум", "политик", "экономик", "движен", "право"],
    }

    DOMAIN_ARCHETYPE_WEIGHTS: dict[str, dict[str, float]] = {
        "urbanism":   {"MDCF": 1.0, "MSCO": 0.8, "ADCO": 0.7, "ADCF": 0.6},
        "biology":    {"MDEF": 1.0, "MSCF": 0.8, "MSEO": 0.6, "ASCO": 0.5},
        "technology": {"ADCO": 1.0, "ADEO": 0.9, "MDCO": 0.8, "ASCO": 0.6},
        "geography":  {"MSCF": 0.9, "MSCO": 0.7, "ASCF": 0.6, "MDCF": 0.5},
        "culture":    {"ASCF": 1.0, "ASEF": 0.9, "ADCF": 0.7, "ASCO": 0.5},
        "science":    {"ASCO": 1.0, "ASEO": 0.9, "ADCO": 0.6, "ADEO": 0.5},
        "society":    {"ADCF": 1.0, "ASCF": 0.8, "MDCF": 0.7, "ADEF": 0.4},
    }

    def parse_topic(self, query: str) -> dict:
        """Извлечь тему, сущности, домен и язык из запроса."""
        entities = self._extract_entities(query)
        domain   = self._classify_domain(query)
        language = "ru" if re.search(r"[а-яА-Я]", query) else "en"
        return {
            "query":    query,
            "entities": entities,
            "domain":   domain,
            "language": language,
        }

    def _extract_entities(self, query: str) -> list[str]:
        words   = query.split()
        # простая эвристика: слова с заглавной буквы (не первое слово)
        entities = [w for w in words[1:] if w and w[0].isupper()]
        return entities if entities else words[:2]

    def _classify_domain(self, query: str) -> str:
        q = query.lower()
        best_domain, best_score = "general", 0
        for domain, keywords in self.DOMAIN_KEYWORDS.items():
            score = sum(1 for kw in keywords if kw.lower() in q)
            if score > best_score:
                best_domain, best_score = domain, score
        return best_domain

    def calculate_archetype_relevance(
        self, topic_info: dict
    ) -> dict[str, float]:
        """Скоринг 16 архетипов для данной темы."""
        domain   = topic_info.get("domain", "general")
        query    = topic_info.get("query", "").lower()
        weights  = self.DOMAIN_ARCHETYPE_WEIGHTS.get(domain, {})

        scores: dict[str, float] = {}
        for a in ARCHETYPES:
            base  = weights.get(a.code, 0.1)
            # бонус за совпадение ключевых слов
            kw_match = sum(1 for kw in a.keywords_ru + a.keywords_en
                           if kw.lower() in query)
            bonus = min(0.5, kw_match * 0.1)
            # бонус за приоритет архетипа
            prio  = a.priority / 5 * 0.2
            scores[a.code] = min(1.0, base + bonus + prio)

        return scores

    def expand_query(self, query: str) -> QuestionTree:
        """
        Главный метод: запрос → QuestionTree.
        Генерирует вопросы по всем релевантным архетипам + 7 Heptagram.
        """
        topic_info  = self.parse_topic(query)
        relevance   = self.calculate_archetype_relevance(topic_info)
        topic       = query

        tree     = QuestionTree(topic=topic)
        q_idx    = 0

        # вопросы по архетипам
        for a in ARCHETYPES:
            score = relevance.get(a.code, 0.0)
            if score < self.MIN_RELEVANCE:
                continue
            templates = QUESTION_TEMPLATES.get(a.code, [])
            for tmpl in templates:
                q_idx += 1
                tree.questions.append(Question(
                    id           = f"Q{q_idx:03d}",
                    text         = tmpl.format(topic=topic),
                    archetype    = a.code,
                    priority     = a.priority,
                    answer_type  = "text",
                ))

        # вопросы Heptagram (7 измерений)
        for dim, tmpl in HEPTAGRAM_QUESTIONS.items():
            q_idx += 1
            tree.questions.append(Question(
                id          = f"H{q_idx:03d}",
                text        = tmpl.format(topic=topic),
                archetype   = "HEPT",
                priority    = 3,
                answer_type = "number",
                dimension   = dim,
            ))

        return tree
