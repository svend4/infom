"""
LLM Adapter — абстракция для языковых моделей.
Поддерживает Ollama, OpenAI-совместимые API, и mock-режим для тестов.
Никаких обязательных зависимостей — всё через stdlib http.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable
import json
import urllib.request
import urllib.error


@dataclass
class LLMResponse:
    text:   str
    model:  str
    tokens: int = 0


class LLMAdapter:
    """
    Базовый адаптер. Может быть заменён любой реализацией.
    """
    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError

    def embed(self, text: str) -> list[float]:
        raise NotImplementedError


class MockLLMAdapter(LLMAdapter):
    """
    Mock-адаптер для тестов и разработки без LLM.
    Использует детерминированные хэши вместо настоящих эмбеддингов.
    """

    # ── фразы-шаблоны для mock-ответов ──────────────────────────────────────
    _INTROS = [
        "На основе графа знаний: {topic} — это {desc}.",
        "{topic} представляет собой {desc}.",
        "Согласно контексту, {topic} — {desc}.",
        "В данной области {topic} понимается как {desc}.",
    ]
    _CONNECTORS = [
        "Ключевые связи: {rels}.",
        "Взаимодействует с: {rels}.",
        "Тесно связан с: {rels}.",
    ]
    _CLOSINGS = [
        "Архетип кластера: {arch}.",
        "Геометрический профиль: {shape}.",
        "Q6-позиция указывает на {arch}-паттерн.",
    ]

    def _mock_answer(self, prompt: str) -> str:
        """Генерирует читаемый текстовый ответ на основе контекста в промпте."""
        import re

        # ── детект global-запроса (содержит сводки по сообществам) ───────────
        if 'Сводка по сообществам' in prompt or re.search(r'\[[\w]+/[\w]+\]', prompt):
            return self._mock_global_answer(prompt)

        # извлекаем тему из строки "Вопрос: ..."
        topic_m = re.search(r'Вопрос:\s*(.+?)(?:\n|$)', prompt)
        topic = topic_m.group(1).strip() if topic_m else "тема"
        q_words = {"что", "как", "где", "зачем", "почему", "кто", "какой",
                   "какие", "такое", "является"}
        topic_words = [w for w in topic.rstrip("?").split()
                       if w.lower() not in q_words]
        topic_short = " ".join(topic_words) if topic_words else topic

        # извлекаем сущности из строки "Сущности: ..."
        ent_m = re.search(r'Сущности:\s*(.+?)(?:\n|$)', prompt)
        entities = [e.strip() for e in ent_m.group(1).split(",")][:4] if ent_m else []

        # связи и архетипы
        rel_m   = re.search(r'Связи:\s*(.+?)(?:\n|$)', prompt)
        arch_m  = re.search(r'Архетипы:\s*(.+?)(?:\n|$)', prompt)
        shape_m = re.search(r'Форма кластера:\s*(\S+)', prompt)

        rels_str  = rel_m.group(1).strip()  if rel_m  else ""
        arch_str  = arch_m.group(1).strip() if arch_m else "ADCO"
        shape_str = shape_m.group(1).strip() if shape_m else "polygon"

        seed = sum(ord(c) for c in topic)
        intro_tmpl = self._INTROS[seed % len(self._INTROS)]
        conn_tmpl  = self._CONNECTORS[(seed // 3) % len(self._CONNECTORS)]
        close_tmpl = self._CLOSINGS[(seed // 7) % len(self._CLOSINGS)]

        desc = entities[0] if entities else "сложная система"
        if len(entities) > 1:
            desc += f", включающий {entities[1]}"

        parts = [intro_tmpl.format(topic=topic_short, desc=desc)]
        if rels_str and rels_str != "нет данных":
            rels_cut = rels_str
            if len(rels_str) > 100:
                idx = rels_str.rfind(";", 0, 100)
                rels_cut = rels_str[:idx] if idx > 0 else rels_str[:100]
            parts.append(conn_tmpl.format(rels=rels_cut))
        parts.append(close_tmpl.format(arch=arch_str, shape=shape_str))

        return " ".join(parts)

    def _mock_global_answer(self, prompt: str) -> str:
        """Синтезирует ответ на global-запрос из community summaries."""
        import re
        topic_m = re.search(r'Вопрос:\s*(.+?)(?:\n|$)', prompt)
        topic   = topic_m.group(1).strip() if topic_m else "граф"

        # парсим строки вида "[shape/arch] Q6=N skel=S dom=D: nodes"
        comm_pattern = re.compile(
            r'\[(\w+)/(\w+)\]\s+Q6=(\d+)[^:]+:\s*([^\n]+)'
        )
        groups = comm_pattern.findall(prompt)
        if not groups:
            return f"Граф охватывает несколько тематических кластеров."

        parts = [f"Граф содержит {len(groups)} сообщества знаний:"]
        arch_names = {
            "ADCO": "программные системы", "ADEO": "алгоритмы",
            "ASCO": "точные науки",        "ASEO": "формальные системы",
            "MDEF": "живые организмы",     "MSCF": "экосистемы",
            "MSCO": "инфраструктура",      "MDCO": "технические системы",
            "MDCF": "социальные сети",     "ADEF": "эвристики",
        }
        for shape, arch, q6, nodes_str in groups[:4]:
            arch_label = arch_names.get(arch, arch)
            nodes = [n.strip() for n in nodes_str.split(",")][:3]
            parts.append(
                f"  • [{shape}/{arch}] {arch_label}: {', '.join(nodes)}"
            )

        # общий вывод
        archs = [g[1] for g in groups]
        domains = sorted(set(arch_names.get(a, a) for a in archs))
        parts.append(f"Основные домены: {'; '.join(domains[:3])}.")
        return "\n".join(parts)

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        # Если промпт запрашивает извлечение сущностей (indexer) → JSON
        if '"entities"' in prompt or 'JSON с полями' in prompt or 'json' in prompt.lower()[:200]:
            # Извлекаем только из раздела "Текст:" если он есть
            text_section = prompt
            if 'Текст:' in prompt:
                text_section = prompt.split('Текст:', 1)[1].strip()
            elif 'текст:' in prompt.lower():
                idx = prompt.lower().index('текст:')
                text_section = prompt[idx + 6:].strip()
            words = [w.strip('.,!?;:()"«»') for w in text_section.split()
                     if len(w) > 3 and w[0].isupper()]
            entities = list(dict.fromkeys(words))[:5]
            if not entities:
                # fallback: любые слова > 4 символов
                words2 = [w.strip('.,!?;:') for w in text_section.split() if len(w) > 4]
                entities = list(dict.fromkeys(words2))[:5]
            result = json.dumps({
                "entities": [{"id": f"e{i}", "label": e, "type": "concept"}
                              for i, e in enumerate(entities)],
                "relations": [{"source": f"e{i}", "target": f"e{i+1}",
                               "label": "связан с", "weight": 0.7}
                               for i in range(len(entities)-1)],
            }, ensure_ascii=False)
            return LLMResponse(text=result, model="mock", tokens=len(prompt)//4)

        # RAG-запрос → читаемый текстовый ответ
        return LLMResponse(
            text=self._mock_answer(prompt),
            model="mock",
            tokens=len(prompt)//4,
        )

    def embed(self, text: str) -> list[float]:
        """Детерминированный псевдо-эмбеддинг через хэш символов."""
        import math
        h = 0
        for ch in text:
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        # 6-мерный вектор из хэша
        vec = []
        for i in range(6):
            h = (h * 1664525 + 1013904223) & 0xFFFFFFFF
            # нормализуем в [-1, 1]
            vec.append((h / 0x7FFFFFFF) - 1.0)
        return vec


class OllamaAdapter(LLMAdapter):
    """
    Адаптер для Ollama (локальный LLM).
    Требует запущенного Ollama: https://ollama.ai
    """

    def __init__(
        self,
        model:       str   = "qwen2.5:14b",
        embed_model: str   = "bge-m3",
        base_url:    str   = "http://localhost:11434",
        timeout:     int   = 60,
    ):
        self.model       = model
        self.embed_model = embed_model
        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        payload = json.dumps({
            "model":  self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs,
        }).encode()

        req  = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data    = payload,
            headers = {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                return LLMResponse(
                    text   = data.get("response", ""),
                    model  = self.model,
                    tokens = data.get("eval_count", 0),
                )
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"Ollama недоступен: {e}") from e

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({
            "model":  self.embed_model,
            "prompt": text,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/api/embeddings",
            data    = payload,
            headers = {"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                return data.get("embedding", [])
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"Ollama недоступен: {e}") from e


class OpenAIAdapter(LLMAdapter):
    """
    Адаптер для OpenAI-совместимых API (OpenAI, LM Studio, vLLM, etc.)
    """

    def __init__(
        self,
        api_key:     str   = "",
        model:       str   = "gpt-4o-mini",
        embed_model: str   = "text-embedding-3-small",
        base_url:    str   = "https://api.openai.com/v1",
        timeout:     int   = 30,
    ):
        self.api_key     = api_key
        self.model       = model
        self.embed_model = embed_model
        self.base_url    = base_url.rstrip("/")
        self.timeout     = timeout

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                text = data["choices"][0]["message"]["content"]
                return LLMResponse(
                    text   = text,
                    model  = self.model,
                    tokens = data.get("usage", {}).get("total_tokens", 0),
                )
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"API недоступен: {e}") from e

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({
            "model": self.embed_model,
            "input": text,
        }).encode()

        req = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                return data["data"][0]["embedding"]
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"API недоступен: {e}") from e
