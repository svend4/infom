"""
Document Indexer — преобразует текстовый документ в граф знаний.

Pipeline:
    Текст
      ↓ chunk()         — разбивка на чанки
      ↓ extract()       — извлечение сущностей и связей (LLM)
      ↓ embed()         — вычисление эмбеддингов
      ↓ classify()      — архетип для каждой сущности
      ↓ build()         — построение KnowledgeMap

Работает с любым LLMAdapter (MockLLM, Ollama, OpenAI).
"""
from __future__ import annotations
from dataclasses import dataclass, field
import json
import re

from graph       import GraphNode, GraphEdge, KnowledgeMap
from llm_adapter import LLMAdapter, MockLLMAdapter
from archetypes  import find_by_keyword, ARCHETYPES


# ── промпты ─────────────────────────────────────────────────────────────────

EXTRACT_PROMPT = """Извлеки из следующего текста:
1. Сущности (entities): имена, понятия, объекты, места, идеи
2. Связи (relations): отношения между сущностями

Верни ТОЛЬКО JSON в формате:
{{
  "entities": [
    {{"id": "e1", "label": "название", "type": "concept|person|place|event|other"}}
  ],
  "relations": [
    {{"source": "e1", "target": "e2", "label": "описание связи", "weight": 0.8}}
  ]
}}

Текст:
{text}
"""


# ── структуры данных ─────────────────────────────────────────────────────────

@dataclass
class Chunk:
    id:      str
    text:    str
    start:   int   # символьная позиция в оригинале
    end:     int


@dataclass
class ExtractedEntity:
    id:       str
    label:    str
    type:     str
    chunk_id: str
    archetype: str = ""


@dataclass
class ExtractedRelation:
    source:   str
    target:   str
    label:    str
    weight:   float
    chunk_id: str


@dataclass
class IndexResult:
    n_chunks:    int
    n_entities:  int
    n_relations: int
    n_nodes:     int
    n_edges:     int
    n_communities: int


# ── чанкинг ─────────────────────────────────────────────────────────────────

def chunk_text(
    text:         str,
    chunk_size:   int = 512,
    overlap:      int = 64,
) -> list[Chunk]:
    """
    Разбивает текст на перекрывающиеся чанки по границам предложений.
    """
    # сначала разбиваем на предложения
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())

    chunks  = []
    current = []
    cur_len = 0
    pos     = 0
    idx     = 0

    for sent in sentences:
        slen = len(sent)
        if cur_len + slen > chunk_size and current:
            chunk_text_str = ' '.join(current)
            chunks.append(Chunk(
                id    = f"chunk_{idx}",
                text  = chunk_text_str,
                start = pos - cur_len,
                end   = pos,
            ))
            idx += 1
            # overlap: оставляем последние предложения
            overlap_sents = []
            overlap_len   = 0
            for s in reversed(current):
                if overlap_len + len(s) > overlap:
                    break
                overlap_sents.insert(0, s)
                overlap_len += len(s)
            current = overlap_sents
            cur_len = overlap_len

        current.append(sent)
        cur_len += slen
        pos     += slen + 1

    if current:
        chunks.append(Chunk(
            id    = f"chunk_{idx}",
            text  = ' '.join(current),
            start = pos - cur_len,
            end   = pos,
        ))

    return chunks if chunks else [Chunk(id="chunk_0", text=text, start=0, end=len(text))]


# ── извлечение сущностей ─────────────────────────────────────────────────────

def _parse_extraction(response_text: str) -> tuple[list[dict], list[dict]]:
    """Парсит JSON из ответа LLM (с защитой от лишнего текста)."""
    # ищем JSON блок
    match = re.search(r'\{.*\}', response_text, re.DOTALL)
    if not match:
        return [], []
    try:
        data = json.loads(match.group())
        return data.get("entities", []), data.get("relations", [])
    except json.JSONDecodeError:
        return [], []


def _guess_archetype(label: str, entity_type: str) -> str:
    """Угадать архетип сущности по метке и типу."""
    matches = find_by_keyword(label)
    if matches:
        return matches[0].code

    type_map = {
        "person":  "MDEF",   # Организм — живое адаптирующееся существо
        "place":   "MSCF",   # Лес — природная/городская среда
        "event":   "MDCF",   # Город — динамическая система
        "concept": "ASCO",   # Теория — структурированное знание
        "other":   "ASEF",   # Архетип — базовый паттерн
    }
    return type_map.get(entity_type, "ASEF")


# ── главный класс ────────────────────────────────────────────────────────────

class DocumentIndexer:
    """
    Индексирует текстовый документ в KnowledgeMap.

    Использование:
        indexer = DocumentIndexer(llm=OllamaAdapter())
        km = indexer.index("Текст документа...")
    """

    def __init__(
        self,
        llm:          LLMAdapter | None = None,
        chunk_size:   int = 512,
        chunk_overlap: int = 64,
        embed_dim:    int = 6,
    ):
        self.llm          = llm or MockLLMAdapter()
        self.chunk_size   = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embed_dim    = embed_dim

    def index(self, text: str, n_communities: int = 6) -> tuple[KnowledgeMap, IndexResult]:
        """
        Полный цикл индексации: текст → KnowledgeMap.
        """
        # 1. чанкинг
        chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)

        # 2. извлечение сущностей и связей
        all_entities: dict[str, ExtractedEntity] = {}
        all_relations: list[ExtractedRelation]   = []

        for chunk in chunks:
            ents, rels = self._extract_chunk(chunk)
            for e in ents:
                # глобальная дедупликация по label (lowercase)
                key = e.label.lower().strip()
                if key not in all_entities:
                    all_entities[key] = e
            all_relations.extend(rels)

        # 3. построение KnowledgeMap
        km = KnowledgeMap()

        # создаём ноды
        label_to_id: dict[str, str] = {}
        for key, ent in all_entities.items():
            embedding = self.llm.embed(ent.label)
            # дополняем до embed_dim если нужно
            while len(embedding) < self.embed_dim:
                embedding.append(0.0)
            embedding = embedding[:self.embed_dim]

            node = GraphNode(
                id        = ent.id,
                label     = ent.label,
                embedding = embedding,
                archetype = ent.archetype,
                metadata  = {"type": ent.type, "chunk": ent.chunk_id},
            )
            km.add_node(node)
            label_to_id[key] = ent.id

        # создаём рёбра
        for rel in all_relations:
            src_key = rel.source.lower().strip()
            tgt_key = rel.target.lower().strip()
            src_id  = label_to_id.get(src_key)
            tgt_id  = label_to_id.get(tgt_key)
            if src_id and tgt_id and src_id != tgt_id:
                km.add_edge(GraphEdge(
                    source   = src_id,
                    target   = tgt_id,
                    label    = rel.label,
                    weight   = rel.weight,
                    directed = True,
                    metadata = {"chunk": rel.chunk_id},
                ))

        # 4. строим граф (community detection, hyper_edges, fractal boundaries)
        km.build(n_communities=n_communities)

        result = IndexResult(
            n_chunks     = len(chunks),
            n_entities   = len(all_entities),
            n_relations  = len(all_relations),
            n_nodes      = len(km.nodes),
            n_edges      = len(km.edges),
            n_communities = len(km.communities),
        )
        return km, result

    def _extract_chunk(
        self, chunk: Chunk
    ) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
        """Извлечь сущности и связи из одного чанка."""
        prompt   = EXTRACT_PROMPT.format(text=chunk.text)
        response = self.llm.complete(prompt)
        raw_ents, raw_rels = _parse_extraction(response.text)

        entities  = []
        id_to_label: dict[str, str] = {}

        for i, e in enumerate(raw_ents):
            eid   = f"{chunk.id}_e{i}"
            label = str(e.get("label", "")).strip()
            etype = str(e.get("type", "concept"))
            if not label:
                continue
            arch = _guess_archetype(label, etype)
            entities.append(ExtractedEntity(
                id        = eid,
                label     = label,
                type      = etype,
                chunk_id  = chunk.id,
                archetype = arch,
            ))
            id_to_label[e.get("id", eid)] = label

        relations = []
        for r in raw_rels:
            src_raw = str(r.get("source", ""))
            tgt_raw = str(r.get("target", ""))
            src_label = id_to_label.get(src_raw, src_raw)
            tgt_label = id_to_label.get(tgt_raw, tgt_raw)
            if not src_label or not tgt_label:
                continue
            relations.append(ExtractedRelation(
                source   = src_label,
                target   = tgt_label,
                label    = str(r.get("label", "связан с")),
                weight   = float(r.get("weight", 0.7)),
                chunk_id = chunk.id,
            ))

        return entities, relations
