---
name: infom-analyze
description: >
  Анализирует текст или граф знаний через InfoM GraphRAG систему.
  Используй когда нужно: построить граф из текста, найти сообщества,
  задать вопрос к графу, оценить семантическую кластеризацию.
allowed-tools: Read, Bash, Glob, Grep
---

# InfoM GraphRAG Analyzer

Ты используешь систему InfoM для анализа знаний. Выполни следующие шаги:

## 1. Определи задачу

- **Индексация текста** → `DocumentIndexer` + `SemanticAdapter`
- **Поиск по графу** → `GraphRAGQuery.query(q, mode='local'|'global'|'hybrid')`
- **Кластеризация** → `KnowledgeMap.build()` + `render_communities()`
- **Семантический тест** → `python test_semantic.py`

## 2. Шаблон кода

```python
from semantic_sim import SemanticAdapter
from indexer import DocumentIndexer
from graphrag_query import GraphRAGQuery
from visualizer.ascii import render_communities, render_graph_ascii

# Индексация
llm = SemanticAdapter()
indexer = DocumentIndexer(llm=llm)
km, result = indexer.index(YOUR_TEXT)
print(f"Граф: {len(km.nodes)} нод, Q={km.modularity:.3f}")
print(render_graph_ascii(km))
print(render_communities(km))

# RAG запросы
rag = GraphRAGQuery(km, llm=llm)
for question in YOUR_QUESTIONS:
    ans = rag.query(question, mode='local')
    print(f"Q: {question}")
    print(f"A: {ans.answer}")
```

## 3. Интерпретация результатов

| Показатель | Хорошо | Плохо |
|---|---|---|
| Modularity Q | > 0.4 | < 0.15 |
| LP iterations | 1-3 | > 5 (unstable) |
| Сообщества | 3-7 | 1 или > N/2 |
| Recall (k=3, r=2) | 74% | < 44% |

## 4. Геометрические формы

- △ triangle = плотная триада (3 понятия)
- □ rectangle = устойчивый паттерн (4)
- ⬠ pentagon = сложный кластер (5+)
- ◇ polygon = малый/нестандартный (< 3)

## 5. Архетипы (4-битовый код)

`[A/M][D/S][C/E][O/F]`
- A=Agentic/M=Morphic, D=Digital/S=Structural
- C=Convergent/E=Exploratory, O=Ordered/F=Fluid

Пример: ADEO = алгоритмический исследовательский (нейросети, ML)
