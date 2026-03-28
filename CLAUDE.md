# InfoM — Geometric GraphRAG System

## О проекте

InfoM — это система GraphRAG (граф знаний + генерация ответов) с геометрической семантикой.
Каждая сущность имеет: 32D embedding, Q6 hex-адрес, TangramSignature, FractalBoundary, HeptagramProfile.

## Архитектура (читай при работе с кодом)

```
llm_adapter.py          — LLM abstraction: MockLLMAdapter, OllamaAdapter, OpenAIAdapter, SemanticAdapter
graph/
  knowledge_map.py      — KnowledgeMap: nodes, edges, communities, build()
  community.py          — Community: dominant_archetype, tangram, heptagram, octagram
indexer.py              — DocumentIndexer: text → chunks → LLM NER → KnowledgeMap
graphrag_query.py       — GraphRAGQuery: local/global/hybrid RAG modes
search/
  hnsw.py               — HNSWSearch: 2-stage search (Q6 Hamming → cosine rerank)
  multi_lsh.py          — MultiProjectionQ6: 3 independent LSH projections
signatures/
  hexsig.py             — Q6 hexagonal space (6-bit, 64 cells, average-pool projection)
  tangram.py            — TangramSignature: geometric shape of community
  fractal.py            — FractalSignature: IFS boundary complexity
archetypes/
  query_expander.py     — 16 archetypes × QueryExpander templates
visualizer/
  ascii.py              — ASCII graph + community rendering
  html.py               — D3.js interactive HTML export
semantic_sim.py         — SemanticAdapter: 32D semantic embeddings (no real LLM needed)
```

## Ключевые числа

- Q6: 6-bit hex space, 64 cells, average-pool projection N→6
- Multi-LSH: 3 projections → 74% recall at radius=2
- Modularity: Mock Q=0.131 vs Semantic32D Q=0.538
- Archetypes: 16 (4 bits: A/M × D/S × C/E × O/F)

## Правила разработки

1. **Embeddings всегда нормированы** (единичная длина) — не нарушай это
2. **Q6 проекция через average-pool** (hexsig.py:embed_to_q6) — не возвращайся к первым 6 dims
3. **Тестируй с SemanticAdapter** — MockLLMAdapter не валидирует семантику
4. **LP threshold = 1.5×** — не снижай, иначе over-merging
5. **HyperEdge score** = avg_node_score × (0.7 + 0.3 × coverage_ratio)

## Как запустить

```bash
python main.py              # полная демонстрация
python test_semantic.py     # тест семантики vs mock
```

## Подключение реального LLM (следующий шаг)

```python
from llm_adapter import OllamaAdapter
llm = OllamaAdapter(model="qwen2.5:14b", embed_model="bge-m3")
# bge-m3 возвращает 1024D — embed_to_q6 автоматически average-pool 1024→6
```
