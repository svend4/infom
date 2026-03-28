# Skill: InfoM GraphRAG

**ID:** `infom-graphrag`
**Version:** 1.0.0
**Author:** svend4
**Category:** knowledge-graph, rag, search
**Tags:** graph, embeddings, semantic-search, knowledge-base, russian, german

## Description

GraphRAG-система с геометрической семантикой. Индексирует тексты в граф знаний (сущности + рёбра + сообщества), отвечает на вопросы через локальный/глобальный/гибридный RAG. Поддерживает накопление знаний между сессиями.

## Base URL

```
https://infom-api.railway.app
```

> Self-hosted: `http://localhost:8000` (запуск: `python infom_api.py`)

## Capabilities

| Команда | Описание |
|---------|----------|
| `проиндексируй [текст]` | Добавить текст в граф знаний |
| `спроси граф: [вопрос]` | Задать вопрос по графу (hybrid режим) |
| `глобальный анализ: [вопрос]` | Анализ всего графа (global режим) |
| `локальный поиск: [вопрос]` | Поиск по ближайшим узлам (local режим) |
| `статистика графа` | Показать метрики графа |
| `сохрани граф` | Сохранить снапшот |
| `загрузи граф` | Восстановить граф из снапшота |
| `сбрось граф` | Очистить граф знаний |
| `визуализация` | Получить HTML с D3.js графом |
| `индексируй и спроси: [текст] :: [вопрос]` | Один запрос: index + query |

## API Reference

### POST /index
Индексировать текст в граф.

```json
{
  "text": "Текст для индексации",
  "reset": false
}
```

**Ответ:**
```json
{
  "ok": true,
  "nodes": 47,
  "edges": 31,
  "communities": 6,
  "modularity": 0.42
}
```

### POST /query
Задать вопрос по графу.

```json
{
  "question": "Что самое важное в документах?",
  "mode": "hybrid"
}
```

Режимы: `local` | `global` | `hybrid`

**Ответ:**
```json
{
  "ok": true,
  "result": "Ответ на основе графа знаний...",
  "mode": "hybrid",
  "graph": {"nodes": 47, "communities": 6, "modularity": 0.42}
}
```

### GET /stats
```json
{
  "nodes": 52,
  "edges": 38,
  "communities": 8,
  "modularity": 0.538,
  "top_communities": [...]
}
```

### POST /webhook
Универсальный endpoint для автоматизации (Make.com, n8n, Zapier):

```json
{
  "action": "index_and_query",
  "text": "{{входящий текст}}",
  "question": "Кратко резюмируй ключевые идеи",
  "mode": "hybrid",
  "reset": false
}
```

Действия: `index` | `query` | `stats` | `reset` | `save` | `load` | `index_and_query`

## Usage Examples

### Пример 1: Анализ документа
```
Пользователь: проиндексируй Решение ФССП от 15.03.2026 об отказе в ИЛ по делу №... [текст решения]
Агент: [вызов POST /index] → Проиндексировано. Граф: 12 узлов, 3 сообщества, Q=0.31

Пользователь: спроси граф: какие нарушения допущены ФССП?
Агент: [вызов POST /query mode=hybrid] → Выявлено 2 нарушения: ...
```

### Пример 2: Накопление базы знаний
```
Пользователь: проиндексируй [текст 1]
Агент: Добавлено. Граф: 8 узлов

Пользователь: проиндексируй [текст 2]
Агент: Добавлено. Граф: 15 узлов

Пользователь: статистика графа
Агент: Граф: 15 узлов, 11 рёбер, 3 сообщества, модулярность 0.41

Пользователь: сохрани граф
Агент: Снапшот сохранён
```

### Пример 3: Глобальный анализ корпуса
```
Пользователь: глобальный анализ: какие главные темы в базе знаний?
Агент: [POST /query mode=global] → В графе 6 тематических кластеров: ...
```

### Пример 4: Одним запросом (webhook)
```
Пользователь: индексируй и спроси: [текст документа] :: что здесь написано?
Агент: [POST /webhook action=index_and_query] → Документ: ..., Ответ: ...
```

## Integration

### Make.com / Zapier / n8n
```
POST https://infom-api.railway.app/webhook
Content-Type: application/json
{
  "action": "index_and_query",
  "text": "{{trigger.data}}",
  "question": "Кратко резюмируй",
  "mode": "hybrid"
}
```

### MCP (прямое подключение)
```json
{
  "mcpServers": {
    "infom": {
      "command": "python",
      "args": ["/path/to/infom/infom_mcp.py"]
    }
  }
}
```

MCP tools: `infom_index`, `infom_query`, `infom_build`, `infom_stats`, `infom_save`, `infom_load`, `infom_reset`, `infom_visualize`, `infom_add_node`, `infom_add_edge`, `infom_benchmark`

## Self-Hosted Deploy

```bash
# Docker
docker run -p 8000:8000 \
  -e INFOM_PROVIDER=groq \
  -e GROQ_API_KEY=gsk-... \
  ghcr.io/svend4/infom:latest

# Python
pip install fastapi uvicorn
INFOM_PROVIDER=semantic python infom_api.py
```

Провайдеры: `semantic` (без ключа) | `groq` | `openrouter` | `cohere` | `jina` | `ollama` | `openai` | `together` | `gemini`

## Source

GitHub: https://github.com/svend4/infom
API docs: `GET /` → JSON с описанием всех endpoints
