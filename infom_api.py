"""
InfoM REST API — HTTP обёртка над MCP инструментами.
Позволяет подключить InfoM к Make.com, n8n, Zapier, Activepieces
и любым другим платформам через стандартные HTTP запросы.

Запуск локально:
  pip install fastapi uvicorn
  python infom_api.py

Запуск с реальным LLM:
  INFOM_PROVIDER=groq GROQ_API_KEY=gsk-... python infom_api.py

API доступен на: http://localhost:8000
Документация:    http://localhost:8000/docs  (Swagger UI)
"""
from __future__ import annotations
import os
import sys
import json
import time
import traceback
from dataclasses import dataclass, asdict

_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

try:
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
    import uvicorn
except ImportError:
    print("Установи зависимости: pip install fastapi uvicorn")
    sys.exit(1)


# ── Импорт MCP инструментов ───────────────────────────────────────────────────

from infom_mcp import (
    tool_index, tool_query, tool_stats, tool_visualize,
    tool_reset, tool_build, tool_save, tool_load,
    tool_add_node, tool_add_edge,
    _state, _get_llm, _get_km,
)


# ── FastAPI приложение ────────────────────────────────────────────────────────

app = FastAPI(
    title       = "InfoM GraphRAG API",
    description = "GraphRAG система с геометрической семантикой. Подключается к Make.com, n8n, Zapier.",
    version     = "1.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Pydantic модели запросов ──────────────────────────────────────────────────

class IndexRequest(BaseModel):
    text:  str
    reset: bool = False

    class Config:
        json_schema_extra = {"example": {
            "text":  "Квантовые вычисления используют кубиты для параллельных вычислений.",
            "reset": False,
        }}

class QueryRequest(BaseModel):
    question: str
    mode:     str = "hybrid"  # local | global | hybrid

    class Config:
        json_schema_extra = {"example": {
            "question": "Что такое квантовое превосходство?",
            "mode":     "hybrid",
        }}

class AddNodeRequest(BaseModel):
    id:        str
    label:     str
    archetype: str   = ""
    weight:    float = 1.0

class AddEdgeRequest(BaseModel):
    source:   str
    target:   str
    label:    str   = "связан"
    weight:   float = 0.7
    directed: bool  = True

class WebhookRequest(BaseModel):
    """Универсальный webhook для Make.com / n8n / Zapier."""
    action:   str            # "index" | "query" | "stats" | "reset"
    text:     str  = ""
    question: str  = ""
    mode:     str  = "hybrid"
    reset:    bool = False
    meta:     dict = {}      # произвольные метаданные от платформы


# ── Эндпоинты ─────────────────────────────────────────────────────────────────

@app.get("/", summary="Статус сервера")
def root():
    km = _state.get("km")
    return {
        "status":   "ok",
        "provider": os.environ.get("INFOM_PROVIDER", "semantic"),
        "graph": {
            "nodes":       len(km.nodes)       if km else 0,
            "edges":       len(km.edges)       if km else 0,
            "communities": len(km.communities) if km else 0,
            "modularity":  round(km.modularity, 3) if km else 0,
        },
    }


@app.post("/index", summary="Индексировать текст в граф знаний")
def index(req: IndexRequest):
    """
    Принимает текст, извлекает сущности и связи, добавляет в граф.
    Используй reset=true для полной очистки перед новой темой.

    **Make.com:** HTTP → POST /index, body = {"text": "...", "reset": false}
    """
    result = tool_index({"text": req.text, "reset": req.reset})
    return {"result": result, "ok": True}


@app.post("/query", summary="Задать вопрос к графу знаний")
def query(req: QueryRequest):
    """
    Выполняет RAG-запрос к графу. Режимы:
    - local  — ближайшие узлы к вопросу
    - global — обзор всех сообществ
    - hybrid — local + global контекст

    **n8n:** HTTP Request → POST /query → поле result в следующую ноду
    """
    km = _state.get("km")
    if not km or not km.nodes:
        raise HTTPException(status_code=400, detail="Граф пуст. Сначала вызови /index")
    result = tool_query({"question": req.question, "mode": req.mode})
    return {"result": result, "ok": True}


@app.get("/stats", summary="Статистика графа")
def stats():
    """Возвращает число нод, рёбер, сообществ, модульность Q, архетипы."""
    return {"result": tool_stats({}), "ok": True}


@app.get("/visualize", summary="ASCII визуализация графа")
def visualize(what: str = "both"):
    """what = graph | communities | both"""
    return {"result": tool_visualize({"what": what}), "ok": True}


@app.post("/reset", summary="Сбросить граф")
def reset():
    return {"result": tool_reset({}), "ok": True}


@app.post("/build", summary="Пересобрать граф после ручных изменений")
def build():
    return {"result": tool_build({}), "ok": True}


@app.post("/save", summary="Сохранить снапшот графа")
def save():
    return {"result": tool_save({}), "ok": True}


@app.post("/load", summary="Загрузить снапшот графа")
def load():
    return {"result": tool_load({}), "ok": True}


@app.post("/node", summary="Добавить узел вручную")
def add_node(req: AddNodeRequest):
    return {"result": tool_add_node(req.model_dump()), "ok": True}


@app.post("/edge", summary="Добавить ребро вручную")
def add_edge(req: AddEdgeRequest):
    return {"result": tool_add_edge(req.model_dump()), "ok": True}


# ── Универсальный Webhook для Make.com / n8n / Zapier ────────────────────────

@app.post("/webhook", summary="Универсальный webhook (Make.com / n8n / Zapier)")
def webhook(req: WebhookRequest):
    """
    Единый endpoint для автоматизационных платформ.

    **Make.com сценарий:**
    1. Trigger (Google Sheets / Notion / RSS)
    2. HTTP POST /webhook body: {"action": "index", "text": "{{content}}"}
    3. HTTP POST /webhook body: {"action": "query", "question": "{{question}}"}
    4. Результат → Slack / Email / Notion

    **Zapier:**
    - Trigger: любой
    - Action: Webhooks by Zapier → POST → /webhook
    - Data: {"action": "query", "question": "{{trigger_field}}"}

    **n8n:**
    - HTTP Request нода → POST /webhook
    - Следующая нода использует {{ $json.result }}
    """
    action = req.action.lower()
    t0 = time.time()

    try:
        if action == "index":
            if not req.text:
                raise HTTPException(status_code=400, detail="text обязателен для action=index")
            result = tool_index({"text": req.text, "reset": req.reset})

        elif action == "query":
            if not req.question:
                raise HTTPException(status_code=400, detail="question обязателен для action=query")
            km = _state.get("km")
            if not km or not km.nodes:
                raise HTTPException(status_code=400, detail="Граф пуст. Сначала action=index")
            result = tool_query({"question": req.question, "mode": req.mode})

        elif action == "stats":
            result = tool_stats({})

        elif action == "reset":
            result = tool_reset({})

        elif action == "save":
            result = tool_save({})

        elif action == "load":
            result = tool_load({})

        elif action == "visualize":
            result = tool_visualize({"what": "both"})

        elif action == "index_and_query":
            # Удобный комбо: одним запросом индексируем и сразу спрашиваем
            if not req.text or not req.question:
                raise HTTPException(status_code=400, detail="text и question обязательны")
            tool_index({"text": req.text, "reset": req.reset})
            result = tool_query({"question": req.question, "mode": req.mode})

        else:
            raise HTTPException(status_code=400, detail=f"Неизвестный action: {action}")

        km = _state.get("km")
        return {
            "ok":      True,
            "action":  action,
            "result":  result,
            "elapsed": round(time.time() - t0, 2),
            "graph": {
                "nodes":       len(km.nodes)       if km else 0,
                "communities": len(km.communities) if km else 0,
                "modularity":  round(km.modularity, 3) if km else 0,
            },
            "meta": req.meta,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"{e}\n{traceback.format_exc()}")


# ── Batch endpoint: индексировать список текстов ──────────────────────────────

@app.post("/batch/index", summary="Пакетная индексация списка текстов")
def batch_index(texts: list[str], reset_first: bool = False):
    """
    Принимает список текстов и индексирует их по очереди.
    Удобно для загрузки нескольких документов одним запросом.

    **Make.com:** Array aggregator → /batch/index
    """
    if reset_first:
        tool_reset({})
    results = []
    for i, text in enumerate(texts):
        r = tool_index({"text": text, "reset": False})
        results.append({"index": i, "ok": True, "preview": r[:100]})
    km = _state.get("km")
    return {
        "ok":           True,
        "indexed":      len(texts),
        "total_nodes":  len(km.nodes) if km else 0,
        "modularity":   round(km.modularity, 3) if km else 0,
        "results":      results,
    }


# ── Запуск ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"InfoM API запущен: http://{host}:{port}")
    print(f"Документация:      http://{host}:{port}/docs")
    uvicorn.run("infom_api:app", host=host, port=port, reload=False)
