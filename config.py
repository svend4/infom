"""
InfoM — конфигурация LLM адаптера.

Выбери провайдера через переменную окружения INFOM_PROVIDER
или измени PROVIDER ниже.

Быстрый старт (бесплатно):
  INFOM_PROVIDER=openrouter OPENROUTER_API_KEY=sk-or-... python infom_mcp.py
  INFOM_PROVIDER=groq       GROQ_API_KEY=gsk-...       python infom_mcp.py
  INFOM_PROVIDER=cohere     COHERE_API_KEY=...          python infom_mcp.py

Для русскоязычного текста (data70) рекомендуется:
  embed: Cohere embed-multilingual-v3.0 или Jina jina-embeddings-v3
  chat:  Groq llama-3.3-70b или OpenRouter gemini-2.0-flash
"""
from __future__ import annotations
import os

# ── Выбор провайдера ─────────────────────────────────────────────────────────
# Варианты: "mock" | "semantic" | "ollama" | "openai" | "cohere" |
#           "groq" | "openrouter" | "jina" | "together" | "gemini"

PROVIDER = os.environ.get("INFOM_PROVIDER", "semantic")

# ── Ключи из переменных окружения ─────────────────────────────────────────────

_KEY = {
    "openai":      os.environ.get("OPENAI_API_KEY",      ""),
    "cohere":      os.environ.get("COHERE_API_KEY",      ""),
    "groq":        os.environ.get("GROQ_API_KEY",        ""),
    "openrouter":  os.environ.get("OPENROUTER_API_KEY",  ""),
    "jina":        os.environ.get("JINA_API_KEY",        ""),
    "together":    os.environ.get("TOGETHER_API_KEY",    ""),
    "gemini":      os.environ.get("GEMINI_API_KEY",      ""),
    "anthropic":   os.environ.get("ANTHROPIC_API_KEY",   ""),
    "ollama_url":  os.environ.get("OLLAMA_URL", "http://localhost:11434"),
}

# ── Настройки по провайдерам ─────────────────────────────────────────────────

OLLAMA_CONFIG = {
    "model":       os.environ.get("OLLAMA_MODEL",       "qwen2.5:14b"),
    "embed_model": os.environ.get("OLLAMA_EMBED_MODEL", "bge-m3"),
    "base_url":    _KEY["ollama_url"],
}

OPENAI_CONFIG = {
    "model":       os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    "embed_model": os.environ.get("OPENAI_EMBED", "text-embedding-3-small"),
}

COHERE_CONFIG = {
    "model":       os.environ.get("COHERE_MODEL", "command-r-plus"),
    "embed_model": os.environ.get("COHERE_EMBED", "embed-multilingual-v3.0"),
    # embed-multilingual-v3.0 → 1024D, лучший для русского
}

GROQ_CONFIG = {
    "model":     os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "embed_via": os.environ.get("GROQ_EMBED_VIA", "jina"),  # jina|cohere|openai
    "embed_key": os.environ.get("GROQ_EMBED_KEY", _KEY["jina"]),
}

OPENROUTER_CONFIG = {
    "model":     os.environ.get("OPENROUTER_MODEL", "google/gemini-2.0-flash-exp:free"),
    "embed_via": os.environ.get("OPENROUTER_EMBED_VIA", "jina"),
    "embed_key": os.environ.get("OPENROUTER_EMBED_KEY", _KEY["jina"]),
}

JINA_CONFIG = {
    "embed_model": os.environ.get("JINA_EMBED", "jina-embeddings-v3"),
}

TOGETHER_CONFIG = {
    "model":       os.environ.get("TOGETHER_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"),
    "embed_model": os.environ.get("TOGETHER_EMBED", "togethercomputer/m2-bert-80M-8k-retrieval"),
}

GEMINI_CONFIG = {
    # Gemini через OpenAI-совместимый endpoint Google
    "model":       os.environ.get("GEMINI_MODEL", "gemini-2.0-flash"),
    "embed_model": os.environ.get("GEMINI_EMBED", "text-embedding-004"),
    "base_url":    "https://generativelanguage.googleapis.com/v1beta/openai",
}


# ── Фабрика адаптеров ─────────────────────────────────────────────────────────

def create_llm_adapter():
    """Создаёт LLM адаптер согласно PROVIDER. Все провайдеры с fallback."""
    provider = PROVIDER.lower()
    print(f"[config] provider={provider}", flush=True)

    try:
        if provider == "mock":
            from llm_adapter import MockLLMAdapter
            return MockLLMAdapter()

        if provider == "semantic":
            from semantic_sim import SemanticAdapter
            return SemanticAdapter()

        if provider == "ollama":
            from llm_adapter import OllamaAdapter
            cfg = OLLAMA_CONFIG
            print(f"[config] Ollama model={cfg['model']} embed={cfg['embed_model']}")
            return OllamaAdapter(**cfg)

        if provider == "openai":
            from llm_adapter import OpenAIAdapter
            _require_key("openai")
            return OpenAIAdapter(
                api_key     = _KEY["openai"],
                model       = OPENAI_CONFIG["model"],
                embed_model = OPENAI_CONFIG["embed_model"],
            )

        if provider == "cohere":
            from llm_adapter import CohereAdapter
            _require_key("cohere")
            return CohereAdapter(
                api_key     = _KEY["cohere"],
                model       = COHERE_CONFIG["model"],
                embed_model = COHERE_CONFIG["embed_model"],
            )

        if provider == "groq":
            from llm_adapter import GroqAdapter
            _require_key("groq")
            return GroqAdapter(
                api_key   = _KEY["groq"],
                model     = GROQ_CONFIG["model"],
                embed_via = GROQ_CONFIG["embed_via"],
                embed_key = GROQ_CONFIG["embed_key"],
            )

        if provider == "openrouter":
            from llm_adapter import OpenRouterAdapter
            _require_key("openrouter")
            return OpenRouterAdapter(
                api_key   = _KEY["openrouter"],
                model     = OPENROUTER_CONFIG["model"],
                embed_via = OPENROUTER_CONFIG["embed_via"],
                embed_key = OPENROUTER_CONFIG["embed_key"],
            )

        if provider == "jina":
            from llm_adapter import JinaAdapter
            _require_key("jina")
            return JinaAdapter(
                api_key     = _KEY["jina"],
                embed_model = JINA_CONFIG["embed_model"],
            )

        if provider == "together":
            from llm_adapter import TogetherAdapter
            _require_key("together")
            return TogetherAdapter(
                api_key     = _KEY["together"],
                model       = TOGETHER_CONFIG["model"],
                embed_model = TOGETHER_CONFIG["embed_model"],
            )

        if provider == "gemini":
            from llm_adapter import OpenAIAdapter
            _require_key("gemini")
            return OpenAIAdapter(
                api_key     = _KEY["gemini"],
                model       = GEMINI_CONFIG["model"],
                embed_model = GEMINI_CONFIG["embed_model"],
                base_url    = GEMINI_CONFIG["base_url"],
            )

        print(f"[config] Неизвестный провайдер '{provider}', fallback → SemanticAdapter")

    except (ValueError, ConnectionError, ImportError) as e:
        print(f"[config] {provider} недоступен: {e}, fallback → SemanticAdapter")

    from semantic_sim import SemanticAdapter
    return SemanticAdapter()


def _require_key(provider: str) -> None:
    if not _KEY.get(provider):
        raise ValueError(
            f"{provider.upper()}_API_KEY не задан. "
            f"Установи переменную окружения или добавь в .env"
        )


# ── Таблица провайдеров (для справки) ────────────────────────────────────────
PROVIDERS_INFO = {
    "mock":        {"embed_dim": 6,    "free": True,  "note": "Детерминированный хэш, без семантики"},
    "semantic":    {"embed_dim": 32,   "free": True,  "note": "Словарный семантический вектор"},
    "ollama":      {"embed_dim": 1024, "free": True,  "note": "Локальный LLM, нужен GPU/CPU"},
    "openai":      {"embed_dim": 1536, "free": False, "note": "gpt-4o-mini + text-embedding-3-small"},
    "cohere":      {"embed_dim": 1024, "free": True,  "note": "Лучший для русского, 1000 embed/мин"},
    "groq":        {"embed_dim": 1024, "free": True,  "note": "Быстрый LPU, embed через Jina/Cohere"},
    "openrouter":  {"embed_dim": 1024, "free": True,  "note": "300+ моделей, есть бесплатные"},
    "jina":        {"embed_dim": 1024, "free": True,  "note": "Только embed, 1M токенов бесплатно"},
    "together":    {"embed_dim": 768,  "free": True,  "note": "$1 кредит, Llama/Qwen"},
    "gemini":      {"embed_dim": 768,  "free": True,  "note": "Щедрый бесплатный тир Google"},
}
