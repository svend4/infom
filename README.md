# InfoM — геометрическая система GraphRAG

Система построения графа знаний и генерации ответов с опорой на граф (**GraphRAG**), использующая геометрическую семантику: каждая сущность представляется нормализованным 32D-вектором, 6-битным адресом в пространстве **Q6** (64 ячейки через average-pool проекцию) и 16 архетипами, а сообщества описываются геометрическими сигнатурами (`TangramSignature`, `FractalSignature`).

## Архитектура и структура проекта

| Модуль / каталог | Назначение |
|---|---|
| `main.py` | Полная демонстрация работы системы (индексация, построение графа, запросы, визуализация) |
| `pipeline.py` | Конвейер обработки документов и построения графа |
| `indexer.py` | `DocumentIndexer`: разбиение текста на чанки, извлечение сущностей (LLM NER) и сборка `KnowledgeMap` |
| `graphrag_query.py` | `GraphRAGQuery`: режимы поиска и генерации ответов (`local`, `global`, `hybrid`) |
| `config.py`, `llm_adapter.py` | Фабрика и адаптеры провайдеров (`mock`, `semantic`, `ollama`, `openai`, `cohere`, `groq`, `openrouter`, `jina`, `together`, `gemini`) через `INFOM_PROVIDER` |
| `semantic_sim.py` | `SemanticAdapter` — локальный 32D-симулятор семантики без обращения к внешнему LLM |
| `graph/` | `knowledge_map.py` (`KnowledgeMap`, Label Propagation, модулярность Ньюмана–Гирвана, PCA-координаты) и `community.py` |
| `search/` | `hnsw.py` (двухэтапный поиск: Q6 Hamming → cosine rerank с бустом по архетипу) и `multi_lsh.py` (3 LSH-проекции) |
| `signatures/` | `hexsig.py` (пространство Q6), `tangram.py` (`TangramSignature`), `fractal.py` (`FractalSignature`) |
| `archetypes/` | `query_expander.py` (шаблоны расширения запросов по 16 архетипам) |
| `visualizer/` | `ascii.py` (ASCII-отрисовка графа и сообществ) и `html.py` (интерактивный D3.js-экспорт в `infom_graph.html`) |
| `infom_mcp.py`, `.mcp.json` | MCP-сервер (stdio JSON-RPC 2.0) с инструментами `infom_index`, `infom_query`, `infom_visualize`, `infom_stats`, `infom_add_node`, `infom_add_edge`, `infom_build`, `infom_reset`, `infom_benchmark`, `infom_save`, `infom_load` |
| `infom_api.py` | FastAPI REST API (`/docs`, `/redoc`, `/webhook`, `/batch/index`, `index_and_query`) |
| `Dockerfile`, `fly.toml`, `railway.toml`, `render.yaml` | Конфигурации контейнеризации и деплоя (Fly.io, Railway, Render) |
| `docs/`, `openclaw-skills/`, `skills/`, `tasks/`, `graph_snapshots/`, `results/` | Архитектурные документы (`genesis.md`, `codex.md`), гайды интеграций (`make_com.md`, `n8n.md`, `zapier_activepieces.md`), навыки и снимки графа |
| `test_semantic.py` | Сравнительные тесты качества эмбеддингов, кластеризации и GraphRAG-запросов |

## Технологический стек и запуск

Ядро системы написано на **чистом Python (stdlib)**; для REST API используются `fastapi` и `uvicorn`, для тестов — `pytest>=7.0.0`, а в `requirements.txt` также перечислены опциональные зависимости для LLM и векторного поиска (`openai`, `anthropic`, `ollama`, `numpy`, `faiss-cpu`, `networkx`, `matplotlib`).

```bash
# Запуск демонстрации и тестов семантики
python main.py
python test_semantic.py

# Запуск MCP-сервера (stdio JSON-RPC 2.0)
python infom_mcp.py

# Запуск FastAPI REST API (http://localhost:8000, Swagger UI: /docs)
pip install fastapi uvicorn
python infom_api.py
# Или с внешним LLM-провайдером:
INFOM_PROVIDER=groq GROQ_API_KEY=gsk-... python infom_api.py
```

```bash
# Сборка и запуск в Docker
docker build -t infom-api .
docker run -p 8000:8000 -e INFOM_PROVIDER=groq -e GROQ_API_KEY=gsk-... infom-api
```

Дополнительные правила разработки и архитектурная карта приведены в [`CLAUDE.md`](CLAUDE.md).

## Ветки репозитория

- `codex/readme-overview-20261118` ([PR #5](../../pull/5), draft) — краткий обзорный `README.md`.
- `claude/wasm-performance-optimization-u8svc` — содержит один неслитый коммит сверх [PR #2](../../pull/2) (`chore: regenerate infom_graph.html`).
- `claude/add-info-section-ZdXes` — изменения уже слиты в `main` через [PR #4](../../pull/4) (ветка оставлена в репозитории).

## Лицензия

Проект распространяется под лицензией **GNU Affero General Public License v3.0** — см. [LICENSE](LICENSE).
