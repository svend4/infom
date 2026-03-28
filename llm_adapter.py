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


class CohereAdapter(LLMAdapter):
    """
    Адаптер для Cohere API.
    Лучший выбор для многоязычных эмбеддингов (русский, немецкий, английский).
    embed-multilingual-v3.0 → 1024D, поддерживает 100+ языков.
    Регистрация: https://cohere.com (бесплатный тир: 1000 embed/мин)
    """

    def __init__(
        self,
        api_key:     str = "",
        model:       str = "command-r-plus",
        embed_model: str = "embed-multilingual-v3.0",
        timeout:     int = 30,
    ):
        self.api_key     = api_key
        self.model       = model
        self.embed_model = embed_model
        self.base_url    = "https://api.cohere.com/v2"
        self.timeout     = timeout

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        payload = json.dumps({
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                text = data["message"]["content"][0]["text"]
                return LLMResponse(
                    text   = text,
                    model  = self.model,
                    tokens = data.get("usage", {}).get("tokens", {}).get("output_tokens", 0),
                )
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"Cohere недоступен: {e}") from e

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({
            "model":      self.embed_model,
            "texts":      [text],
            "input_type": "search_document",
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/embed",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read())
                return data["embeddings"]["float"][0]
        except (urllib.error.URLError, OSError) as e:
            raise ConnectionError(f"Cohere недоступен: {e}") from e


class JinaAdapter(LLMAdapter):
    """
    Адаптер для Jina AI — специализированный на эмбеддингах.
    jina-embeddings-v3 → 1024D, мультиязычный, SOTA на MTEB.
    Бесплатный тир: 1 000 000 токенов.
    Регистрация: https://jina.ai
    """

    def __init__(
        self,
        api_key:     str = "",
        embed_model: str = "jina-embeddings-v3",
        timeout:     int = 30,
    ):
        self.api_key     = api_key
        self.embed_model = embed_model
        self.timeout     = timeout

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        from semantic_sim import SemanticAdapter
        return SemanticAdapter().complete(prompt, **kwargs)

    def embed(self, text: str) -> list[float]:
        payload = json.dumps({
            "model": self.embed_model,
            "input": [text],
            "task":  "retrieval.passage",
        }).encode()
        req = urllib.request.Request(
            "https://api.jina.ai/v1/embeddings",
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
            raise ConnectionError(f"Jina недоступен: {e}") from e


class GroqAdapter(OpenAIAdapter):
    """
    Адаптер для Groq — сверхбыстрый инференс на LPU чипах.
    Бесплатный тир: 14 400 запросов/день.
    Регистрация: https://console.groq.com
    Модели: llama-3.3-70b-versatile, mixtral-8x7b, gemma2-9b-it
    Groq не делает эмбеддинги — нужен отдельный embed_adapter.
    """

    def __init__(
        self,
        api_key:      str            = "",
        model:        str            = "llama-3.3-70b-versatile",
        embed_via:    str            = "jina",    # "jina" | "cohere" | "openai"
        embed_key:    str            = "",
        timeout:      int            = 30,
    ):
        super().__init__(
            api_key  = api_key,
            model    = model,
            base_url = "https://api.groq.com/openai/v1",
            timeout  = timeout,
        )
        self._embed_via = embed_via
        self._embed_key = embed_key or api_key
        self._embed_adapter: LLMAdapter | None = None

    def _get_embed_adapter(self) -> LLMAdapter:
        if self._embed_adapter is None:
            if self._embed_via == "cohere":
                self._embed_adapter = CohereAdapter(api_key=self._embed_key)
            elif self._embed_via == "openai":
                self._embed_adapter = OpenAIAdapter(api_key=self._embed_key)
            else:
                self._embed_adapter = JinaAdapter(api_key=self._embed_key)
        return self._embed_adapter

    def embed(self, text: str) -> list[float]:
        return self._get_embed_adapter().embed(text)


class OpenRouterAdapter(OpenAIAdapter):
    """
    Адаптер для OpenRouter — агрегатор 300+ моделей через один API.
    Бесплатные модели: google/gemini-2.0-flash-exp:free, meta-llama/llama-3.1-8b-instruct:free
    Регистрация: https://openrouter.ai
    """

    def __init__(
        self,
        api_key:     str = "",
        model:       str = "google/gemini-2.0-flash-exp:free",
        embed_via:   str = "jina",    # "jina" | "cohere" | "openai"
        embed_key:   str = "",
        timeout:     int = 30,
    ):
        super().__init__(
            api_key  = api_key,
            model    = model,
            base_url = "https://openrouter.ai/api/v1",
            timeout  = timeout,
        )
        self._embed_via = embed_via
        self._embed_key = embed_key or api_key
        self._embed_adapter: LLMAdapter | None = None

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        payload = json.dumps({
            "model":    self.model,
            "messages": [{"role": "user", "content": prompt}],
            **kwargs,
        }).encode()
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data    = payload,
            headers = {
                "Content-Type":  "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer":  "https://github.com/svend4/infom",
                "X-Title":       "InfoM GraphRAG",
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
            raise ConnectionError(f"OpenRouter недоступен: {e}") from e

    def embed(self, text: str) -> list[float]:
        if self._embed_adapter is None:
            if self._embed_via == "cohere":
                self._embed_adapter = CohereAdapter(api_key=self._embed_key)
            elif self._embed_via == "openai":
                self._embed_adapter = OpenAIAdapter(api_key=self._embed_key)
            else:
                self._embed_adapter = JinaAdapter(api_key=self._embed_key)
        return self._embed_adapter.embed(text)


class TogetherAdapter(OpenAIAdapter):
    """
    Адаптер для Together AI — дешёвый хостинг open-source моделей.
    Бесплатно: $1 стартовый кредит. Embeddings: m2-bert → 768D.
    Регистрация: https://together.ai
    """

    def __init__(
        self,
        api_key:     str = "",
        model:       str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        embed_model: str = "togethercomputer/m2-bert-80M-8k-retrieval",
        timeout:     int = 30,
    ):
        super().__init__(
            api_key     = api_key,
            model       = model,
            embed_model = embed_model,
            base_url    = "https://api.together.xyz/v1",
            timeout     = timeout,
        )
