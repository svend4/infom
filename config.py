"""
InfoM — конфигурация LLM адаптера.

Выбери провайдера и раскомментируй нужный блок.
Затем запусти: python start_server.py

Без реального LLM система использует SemanticAdapter (32D embeddings, Mock-генерация).
С реальным LLM качество ответов значительно улучшается.
"""
from __future__ import annotations
import os


# ── Выбор провайдера ─────────────────────────────────────────────────────────
# Варианты: "mock", "semantic", "ollama", "openai", "anthropic"

PROVIDER = os.environ.get("INFOM_PROVIDER", "semantic")


# ── Настройки по провайдерам ─────────────────────────────────────────────────

OLLAMA_CONFIG = {
    "model":       os.environ.get("OLLAMA_MODEL",       "qwen2.5:14b"),
    "embed_model": os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3"),
    "base_url":    os.environ.get("OLLAMA_URL",         "http://localhost:11434"),
    # bge-m3 возвращает 1024D → Q6 average-pool автоматически
}

OPENAI_CONFIG = {
    "api_key":     os.environ.get("OPENAI_API_KEY", ""),
    "model":       os.environ.get("OPENAI_MODEL",   "gpt-4o-mini"),
    "embed_model": os.environ.get("OPENAI_EMBED",   "text-embedding-3-small"),
    # text-embedding-3-small → 1536D → average-pool → Q6
}

ANTHROPIC_CONFIG = {
    "api_key":     os.environ.get("ANTHROPIC_API_KEY", ""),
    "model":       os.environ.get("ANTHROPIC_MODEL",   "claude-haiku-4-5-20251001"),
    # Для embeddings Anthropic использует OpenAI или локальную модель
    "embed_via":   os.environ.get("ANTHROPIC_EMBED_VIA", "openai"),
}


# ── Фабрика адаптеров ────────────────────────────────────────────────────────

def create_llm_adapter():
    """Создаёт LLM адаптер согласно конфигурации PROVIDER."""
    provider = PROVIDER.lower()

    if provider == "mock":
        from llm_adapter import MockLLMAdapter
        return MockLLMAdapter()

    if provider == "semantic":
        from semantic_sim import SemanticAdapter
        return SemanticAdapter()

    if provider == "ollama":
        try:
            from llm_adapter import OllamaAdapter
            cfg = OLLAMA_CONFIG
            print(f"[config] Ollama: model={cfg['model']}, embed={cfg['embed_model']}")
            return OllamaAdapter(
                model=cfg["model"],
                embed_model=cfg["embed_model"],
                base_url=cfg["base_url"],
            )
        except ImportError as e:
            print(f"[config] Ollama недоступен ({e}), fallback → SemanticAdapter")
            from semantic_sim import SemanticAdapter
            return SemanticAdapter()

    if provider == "openai":
        try:
            from llm_adapter import OpenAIAdapter
            cfg = OPENAI_CONFIG
            if not cfg["api_key"]:
                raise ValueError("OPENAI_API_KEY не задан")
            print(f"[config] OpenAI: model={cfg['model']}, embed={cfg['embed_model']}")
            return OpenAIAdapter(
                api_key=cfg["api_key"],
                model=cfg["model"],
                embed_model=cfg["embed_model"],
            )
        except (ImportError, ValueError) as e:
            print(f"[config] OpenAI недоступен ({e}), fallback → SemanticAdapter")
            from semantic_sim import SemanticAdapter
            return SemanticAdapter()

    # Неизвестный провайдер — безопасный fallback
    print(f"[config] Неизвестный провайдер '{provider}', используем SemanticAdapter")
    from semantic_sim import SemanticAdapter
    return SemanticAdapter()
