#!/usr/bin/env python3
"""
InfoM MCP Server — Model Context Protocol сервер для GraphRAG системы.

Протокол: JSON-RPC 2.0 over stdio (стандарт MCP).
Транспорт: stdin/stdout, каждое сообщение — одна строка JSON.

Инструменты (tools):
  infom_index      — индексировать текст → граф знаний
  infom_query      — задать вопрос к графу (local/global/hybrid)
  infom_visualize  — ASCII визуализация графа и сообществ
  infom_stats      — статистика: нодов, рёбер, Q, архетипы
  infom_add_node   — добавить узел вручную
  infom_add_edge   — добавить ребро вручную
  infom_build      — пересобрать граф (после ручного добавления)
  infom_reset      — сбросить граф
  infom_benchmark  — recall benchmark Multi-Proj Q6 LSH

Запуск:
  python infom_mcp.py

Конфигурация (.mcp.json в корне проекта):
  {
    "mcpServers": {
      "infom": {
        "command": "python",
        "args": ["/path/to/infom/infom_mcp.py"]
      }
    }
  }
"""
from __future__ import annotations
import sys
import json
import os
import traceback
from datetime import datetime

# Добавляем корень проекта в sys.path
_DIR = os.path.dirname(os.path.abspath(__file__))
if _DIR not in sys.path:
    sys.path.insert(0, _DIR)

# ── Персистентность графа ─────────────────────────────────────────────────────

_SNAPSHOTS_DIR = os.path.join(_DIR, "graph_snapshots")
_SNAPSHOT_PATH = os.path.join(_SNAPSHOTS_DIR, "latest.json")


def _save_snapshot() -> None:
    """Сохраняет текущий граф в graph_snapshots/latest.json."""
    km = _state.get("km")
    if km is None or not km.nodes:
        return
    os.makedirs(_SNAPSHOTS_DIR, exist_ok=True)
    data = {
        "saved_at": datetime.now().isoformat(),
        "nodes": [
            {
                "id": n.id,
                "label": n.label,
                "archetype": n.archetype,
                "weight": n.weight,
                "embedding": list(n.embedding),
                "metadata": n.metadata,
            }
            for n in km.nodes.values()
        ],
        "edges": [
            {
                "source": e.source,
                "target": e.target,
                "label": e.label,
                "weight": e.weight,
                "directed": e.directed,
            }
            for e in km.edges
        ],
    }
    with open(_SNAPSHOT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[infom] snapshot saved → {_SNAPSHOT_PATH}", file=sys.stderr)


def _load_snapshot() -> bool:
    """Загружает граф из graph_snapshots/latest.json. Возвращает True при успехе."""
    if not os.path.exists(_SNAPSHOT_PATH):
        return False
    try:
        with open(_SNAPSHOT_PATH, encoding="utf-8") as f:
            data = json.load(f)
        from graph import KnowledgeMap, GraphNode, GraphEdge
        km = KnowledgeMap()
        llm = _get_llm()
        for nd in data.get("nodes", []):
            raw_emb = nd.get("embedding")
            emb = raw_emb if raw_emb else llm.embed(nd["label"])
            node = GraphNode(
                id=nd["id"],
                label=nd["label"],
                archetype=nd.get("archetype", ""),
                weight=nd.get("weight", 1.0),
                embedding=emb,
                metadata=nd.get("metadata", {}),
            )
            km.add_node(node)

        # Нормализуем размерность эмбеддингов: все к max_dim (padding нулями)
        all_dims = [len(n.embedding) for n in km.nodes.values() if n.embedding]
        if all_dims:
            max_dim = max(all_dims)
            for n in km.nodes.values():
                if len(n.embedding) < max_dim:
                    n.embedding = list(n.embedding) + [0.0] * (max_dim - len(n.embedding))

        for ed in data.get("edges", []):
            # пропускаем рёбра с отсутствующими нодами
            if ed["source"] not in km.nodes or ed["target"] not in km.nodes:
                continue
            edge = GraphEdge(
                source=ed["source"],
                target=ed["target"],
                label=ed.get("label", "связан"),
                weight=ed.get("weight", 0.7),
                directed=ed.get("directed", True),
            )
            km.add_edge(edge)
        km.build()
        _state["km"] = km
        _state["rag"] = None
        saved_at = data.get("saved_at", "?")
        print(f"[infom] snapshot loaded ({len(km.nodes)} nodes, saved {saved_at})", file=sys.stderr)
        return True
    except Exception as e:
        print(f"[infom] snapshot load failed: {e}", file=sys.stderr)
        return False


# ── Глобальное состояние ─────────────────────────────────────────────────────

_state: dict = {
    "km":  None,   # KnowledgeMap
    "rag": None,   # GraphRAGQuery
    "llm": None,   # LLMAdapter
}

def _get_llm():
    if _state["llm"] is None:
        from config import create_llm_adapter
        _state["llm"] = create_llm_adapter()
    return _state["llm"]

def _get_km():
    from graph import KnowledgeMap
    if _state["km"] is None:
        _state["km"] = KnowledgeMap()
    return _state["km"]

def _get_rag():
    from graphrag_query import GraphRAGQuery
    km = _get_km()
    if _state["rag"] is None or _state["rag"].km is not km:
        _state["rag"] = GraphRAGQuery(km, llm=_get_llm())
    return _state["rag"]


# ── Описание инструментов ────────────────────────────────────────────────────

TOOLS = [
    {
        "name": "infom_index",
        "description": (
            "Индексирует текст в граф знаний InfoM. "
            "Извлекает сущности и связи, строит Q6-кластеры с геометрическими сигнатурами. "
            "Возвращает статистику графа: число нод, рёбер, сообществ, модульность Q."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "Текст для индексации (русский или английский)"
                },
                "reset": {
                    "type": "boolean",
                    "description": "Если true — сбросить текущий граф перед индексацией",
                    "default": False
                }
            },
            "required": ["text"]
        }
    },
    {
        "name": "infom_query",
        "description": (
            "Задаёт вопрос к графу знаний через GraphRAG. "
            "Режимы: local (вокруг конкретных нод), global (все сообщества), "
            "hybrid (local + global). "
            "Возвращает ответ, источники и ключевые уточняющие вопросы."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Вопрос к графу знаний"
                },
                "mode": {
                    "type": "string",
                    "enum": ["local", "global", "hybrid"],
                    "description": "Режим поиска: local/global/hybrid",
                    "default": "local"
                }
            },
            "required": ["question"]
        }
    },
    {
        "name": "infom_visualize",
        "description": (
            "Возвращает ASCII визуализацию графа и сообществ. "
            "Включает: PCA-граф, таблицу сообществ с архетипами/формами, "
            "гиперрёбра, фрактальные границы, Q6 карту."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "what": {
                    "type": "string",
                    "enum": ["graph", "communities", "both"],
                    "description": "Что визуализировать",
                    "default": "both"
                }
            }
        }
    },
    {
        "name": "infom_stats",
        "description": (
            "Возвращает подробную статистику текущего графа знаний: "
            "число нод/рёбер/сообществ, модульность Q, LP итерации, "
            "распределение архетипов, топ-сообщества по размеру."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "infom_add_node",
        "description": (
            "Добавляет узел в граф вручную. "
            "Эмбеддинг вычисляется автоматически из метки. "
            "После добавления нужно вызвать infom_build() для перестройки индексов."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "id": {
                    "type": "string",
                    "description": "Уникальный идентификатор узла"
                },
                "label": {
                    "type": "string",
                    "description": "Название/метка узла (используется для эмбеддинга)"
                },
                "archetype": {
                    "type": "string",
                    "description": "Архетип (4 символа, например ADEO, MSCF). Опционально.",
                    "default": ""
                },
                "weight": {
                    "type": "number",
                    "description": "Вес узла (0.0-1.0)",
                    "default": 1.0
                }
            },
            "required": ["id", "label"]
        }
    },
    {
        "name": "infom_add_edge",
        "description": (
            "Добавляет ребро между двумя узлами. "
            "После добавления нужно вызвать infom_build() для перестройки индексов."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "ID исходного узла"
                },
                "target": {
                    "type": "string",
                    "description": "ID целевого узла"
                },
                "label": {
                    "type": "string",
                    "description": "Описание связи",
                    "default": "связан"
                },
                "weight": {
                    "type": "number",
                    "description": "Вес ребра (0.0-1.0)",
                    "default": 0.7
                },
                "directed": {
                    "type": "boolean",
                    "description": "Направленное ли ребро",
                    "default": True
                }
            },
            "required": ["source", "target"]
        }
    },
    {
        "name": "infom_build",
        "description": (
            "Перестраивает граф: Q6 Voronoi → Label Propagation → "
            "модульность → гиперрёбра → фрактальные границы. "
            "Вызывать после ручного добавления нод/рёбер через add_node/add_edge."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "infom_reset",
        "description": "Сбрасывает граф знаний (все ноды, рёбра, сообщества).",
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "infom_benchmark",
        "description": (
            "Запускает benchmark Multi-Projection Q6 LSH. "
            "Показывает recall при 1/2/3/5 проекциях на radius=1 и radius=2. "
            "Полезно для понимания качества поиска без реального LLM."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "n_vectors": {
                    "type": "integer",
                    "description": "Число тестовых векторов",
                    "default": 200
                }
            }
        }
    },
    {
        "name": "infom_save",
        "description": (
            "Сохраняет текущий граф знаний в graph_snapshots/latest.json. "
            "Снапшот автоматически загружается при следующем запуске сервера. "
            "Используй после важных изменений или в конце рабочей сессии."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
    {
        "name": "infom_load",
        "description": (
            "Загружает граф из последнего снапшота (graph_snapshots/latest.json). "
            "Позволяет восстановить граф после перезапуска сервера. "
            "Возвращает статистику восстановленного графа."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {}
        }
    },
]


# ── Реализация инструментов ───────────────────────────────────────────────────

def tool_index(args: dict) -> str:
    text  = args.get("text", "").strip()
    reset = args.get("reset", False)
    if not text:
        return "Ошибка: текст не передан."

    if reset:
        _state["km"]  = None
        _state["rag"] = None

    from indexer import DocumentIndexer
    llm     = _get_llm()
    indexer = DocumentIndexer(llm=llm)
    km, result = indexer.index(text)
    _state["km"]  = km
    _state["rag"] = None  # сбросим RAG — пересоздастся с новым km
    _save_snapshot()

    lines = [
        f"Граф успешно проиндексирован.",
        f"",
        f"Статистика:",
        f"  Чанков:      {result.n_chunks}",
        f"  Сущностей:   {result.n_entities}",
        f"  Связей:      {result.n_relations}",
        f"  Нод в графе: {len(km.nodes)}",
        f"  Рёбер:       {len(km.edges)}",
        f"  Сообществ:   {len(km.communities)}",
        f"  Модульность: Q={km.modularity:.3f}",
        f"  LP итераций: {km.lp_iterations}",
        f"",
        f"Сообщества:",
    ]
    for c in km.communities.values():
        nodes_str = ", ".join(n.label for n in c.nodes[:4])
        if len(c.nodes) > 4:
            nodes_str += f" +{len(c.nodes)-4}"
        shape = c.tangram.shape_class.value if c.tangram else "?"
        lines.append(f"  [{c.dominant_archetype}] {shape:10s}  {nodes_str}")

    return "\n".join(lines)


def tool_query(args: dict) -> str:
    question = args.get("question", "").strip()
    mode     = args.get("mode", "local")
    if not question:
        return "Ошибка: вопрос не передан."

    km = _get_km()
    if not km.nodes:
        return "Граф пуст. Сначала вызовите infom_index() для индексации текста."

    rag = _get_rag()
    ans = rag.query(question, mode=mode)

    lines = [
        f"Вопрос [{mode}]: {question}",
        f"",
        f"Ответ:",
        ans.answer,
        f"",
        f"Источники: {', '.join(ans.sources[:5])}",
        f"Токены:    {ans.tokens_used}",
    ]

    if ans.questions and ans.questions.questions:
        lines.append("")
        lines.append("Уточняющие вопросы:")
        for q in ans.questions.questions[:3]:
            arch = getattr(q, "archetype", getattr(q, "archetype_code", ""))
            lines.append(f"  [{arch}] {q.text}")

    return "\n".join(lines)


def tool_visualize(args: dict) -> str:
    what = args.get("what", "both")
    km   = _get_km()
    if not km.nodes:
        return "Граф пуст. Сначала вызовите infom_index()."

    from visualizer.ascii import render_graph_ascii, render_communities
    parts = []
    if what in ("graph", "both"):
        parts.append(render_graph_ascii(km))
    if what in ("communities", "both"):
        parts.append(render_communities(km))
    return "\n".join(parts)


def tool_stats(args: dict) -> str:
    km = _get_km()
    if not km.nodes:
        return "Граф пуст."

    # Распределение архетипов
    arch_count: dict[str, int] = {}
    for node in km.nodes.values():
        arch_count[node.archetype] = arch_count.get(node.archetype, 0) + 1

    # Топ-сообщества
    comms_sorted = sorted(
        km.communities.values(),
        key=lambda c: len(c.nodes), reverse=True
    )

    lines = [
        "Статистика графа InfoM:",
        f"",
        f"  Нод:           {len(km.nodes)}",
        f"  Рёбер:         {len(km.edges)}",
        f"  Гиперрёбер:    {len(km.hyper_edges)}",
        f"  Сообществ:     {len(km.communities)}",
        f"  Границ:        {len(km.borders)}",
        f"  Модульность Q: {km.modularity:.3f}",
        f"  LP итераций:   {km.lp_iterations}",
        f"",
        f"Архетипы нод:",
    ]
    for arch, cnt in sorted(arch_count.items(), key=lambda x: -x[1]):
        bar = "█" * cnt + "░" * max(0, 8 - cnt)
        lines.append(f"  {arch}  [{bar}] {cnt}")

    lines += ["", "Топ-5 сообществ:"]
    for c in comms_sorted[:5]:
        nodes_str = ", ".join(n.label for n in c.nodes[:3])
        if len(c.nodes) > 3:
            nodes_str += f" +{len(c.nodes)-3}"
        shape = c.tangram.shape_class.value if c.tangram else "?"
        lines.append(
            f"  [{c.dominant_archetype}] {shape:10s}  "
            f"Q6={c.hex_id:2d}  {len(c.nodes)} нод  — {nodes_str}"
        )

    return "\n".join(lines)


def tool_add_node(args: dict) -> str:
    nid   = args.get("id", "").strip()
    label = args.get("label", "").strip()
    arch  = args.get("archetype", "")
    w     = float(args.get("weight", 1.0))
    if not nid or not label:
        return "Ошибка: id и label обязательны."

    from graph import GraphNode
    llm   = _get_llm()
    emb   = llm.embed(label)
    km    = _get_km()
    node  = GraphNode(id=nid, label=label, archetype=arch, embedding=emb, weight=w)
    km.add_node(node)
    _state["rag"] = None
    return f"Узел добавлен: {nid} ({label}), hex_id будет вычислен после infom_build()."


def tool_add_edge(args: dict) -> str:
    src      = args.get("source", "").strip()
    tgt      = args.get("target", "").strip()
    label    = args.get("label", "связан")
    weight   = float(args.get("weight", 0.7))
    directed = bool(args.get("directed", True))
    if not src or not tgt:
        return "Ошибка: source и target обязательны."

    from graph import GraphEdge
    km = _get_km()
    if src not in km.nodes:
        return f"Ошибка: узел '{src}' не найден в графе."
    if tgt not in km.nodes:
        return f"Ошибка: узел '{tgt}' не найден в графе."

    km.add_edge(GraphEdge(source=src, target=tgt, label=label,
                          weight=weight, directed=directed))
    _state["rag"] = None
    return f"Ребро добавлено: {src} —{label}→ {tgt} (вес={weight:.2f})."


def tool_build(args: dict) -> str:
    km = _get_km()
    if not km.nodes:
        return "Граф пуст — нечего строить."
    km.build()
    _state["rag"] = None
    _save_snapshot()
    return (
        f"Граф перестроен: {len(km.nodes)} нод, {len(km.edges)} рёбер, "
        f"{len(km.communities)} сообществ, Q={km.modularity:.3f}, "
        f"LP={km.lp_iterations} iter."
    )


def tool_reset(args: dict) -> str:
    _state["km"]  = None
    _state["rag"] = None
    return "Граф сброшен."


def tool_save(args: dict) -> str:
    km = _state.get("km")
    if km is None or not km.nodes:
        return "Граф пуст — нечего сохранять."
    _save_snapshot()
    return (
        f"Граф сохранён → {_SNAPSHOT_PATH}\n"
        f"Нод: {len(km.nodes)}, Рёбер: {len(km.edges)}, "
        f"Сообществ: {len(km.communities)}, Q={km.modularity:.3f}"
    )


def tool_load(args: dict) -> str:
    ok = _load_snapshot()
    if not ok:
        return f"Снапшот не найден: {_SNAPSHOT_PATH}\nСначала вызовите infom_index() или infom_save()."
    km = _state["km"]
    return (
        f"Граф восстановлен из снапшота.\n"
        f"Нод: {len(km.nodes)}, Рёбер: {len(km.edges)}, "
        f"Сообществ: {len(km.communities)}, Q={km.modularity:.3f}"
    )


def tool_benchmark(args: dict) -> str:
    n = int(args.get("n_vectors", 200))
    from search.benchmark import run_recall_benchmark
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        run_recall_benchmark(n_vectors=n, radius=2)
    return buf.getvalue().strip()


_TOOL_HANDLERS = {
    "infom_index":     tool_index,
    "infom_query":     tool_query,
    "infom_visualize": tool_visualize,
    "infom_stats":     tool_stats,
    "infom_add_node":  tool_add_node,
    "infom_add_edge":  tool_add_edge,
    "infom_build":     tool_build,
    "infom_reset":     tool_reset,
    "infom_benchmark": tool_benchmark,
    "infom_save":      tool_save,
    "infom_load":      tool_load,
}


def call_tool(name: str, args: dict) -> dict:
    """Вызывает инструмент по имени. Возвращает MCP content dict."""
    handler = _TOOL_HANDLERS.get(name)
    if handler is None:
        return {
            "content": [{"type": "text", "text": f"Неизвестный инструмент: {name}"}],
            "isError": True,
        }
    try:
        result = handler(args)
        return {
            "content": [{"type": "text", "text": result}],
            "isError": False,
        }
    except Exception as e:
        tb = traceback.format_exc()
        return {
            "content": [{"type": "text", "text": f"Ошибка: {e}\n{tb}"}],
            "isError": True,
        }


# ── JSON-RPC 2.0 / MCP Protocol ──────────────────────────────────────────────

def make_response(id_, result) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def make_error(id_, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": id_, "error": {"code": code, "message": message}}


def handle_message(msg: dict) -> dict | None:
    method = msg.get("method", "")
    id_    = msg.get("id")
    params = msg.get("params") or {}

    # Уведомления — нет ответа
    if id_ is None and method.startswith("notifications/"):
        return None

    if method == "initialize":
        return make_response(id_, {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {"listChanged": False},
            },
            "serverInfo": {
                "name":    "infom-mcp",
                "version": "1.0.0",
            },
        })

    if method == "tools/list":
        return make_response(id_, {"tools": TOOLS})

    if method == "tools/call":
        name = params.get("name", "")
        args = params.get("arguments") or {}
        result = call_tool(name, args)
        return make_response(id_, result)

    if method == "ping":
        return make_response(id_, {})

    # Неизвестный метод
    return make_error(id_, -32601, f"Method not found: {method}")


# ── Главный цикл ─────────────────────────────────────────────────────────────

def main():
    # Все диагностические сообщения — в stderr, не в stdout
    print("InfoM MCP Server запущен. Ожидание сообщений...", file=sys.stderr, flush=True)
    # Автозагрузка снапшота при старте
    _load_snapshot()

    for raw_line in sys.stdin:
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        try:
            msg = json.loads(raw_line)
        except json.JSONDecodeError as e:
            err = make_error(None, -32700, f"Parse error: {e}")
            print(json.dumps(err, ensure_ascii=False), flush=True)
            continue

        response = handle_message(msg)
        if response is not None:
            print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
