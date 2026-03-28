"""
GraphNode — сущность в графе знаний с геометрическими подписями.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

from signatures import (
    HexSignature, build_hex_signature,
    TangramSignature, FractalSignature,
    HeptagramSignature, OctagramSignature,
)


@dataclass
class GraphNode:
    """
    Сущность / понятие в графе знаний.
    Несёт embedding + Q6-позицию + метаданные.
    """
    id:          str
    label:       str
    embedding:   list[float]
    archetype:   str = ""          # один из 16 архетипов pseudorag
    hex_sig:     HexSignature | None = None
    weight:      float = 1.0
    metadata:    dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.hex_sig is None and self.embedding:
            self.hex_sig = build_hex_signature(self.embedding)

    @property
    def hex_id(self) -> int:
        return self.hex_sig.hex_id if self.hex_sig else 0


@dataclass
class GraphEdge:
    """
    Бинарное ребро между двумя нодами (уровень 1).
    """
    source:     str
    target:     str
    label:      str
    weight:     float = 1.0
    directed:   bool  = True
    metadata:   dict[str, Any] = field(default_factory=dict)
