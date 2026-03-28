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

    def complete(self, prompt: str, **kwargs) -> LLMResponse:
        # простейший mock: извлекаем слова которые выглядят как сущности
        words = [w.strip('.,!?;:') for w in prompt.split()
                 if len(w) > 3 and w[0].isupper()]
        entities = list(dict.fromkeys(words))[:5]
        relations = []
        for i in range(len(entities) - 1):
            relations.append(f'{entities[i]} → связан с → {entities[i+1]}')

        result = json.dumps({
            "entities": [{"id": f"e{i}", "label": e, "type": "concept"}
                         for i, e in enumerate(entities)],
            "relations": [{"source": f"e{i}", "target": f"e{i+1}",
                           "label": "связан с", "weight": 0.7}
                          for i in range(len(entities)-1)],
        }, ensure_ascii=False)
        return LLMResponse(text=result, model="mock", tokens=len(prompt)//4)

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
