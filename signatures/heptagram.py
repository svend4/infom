"""
HeptagramSignature — 7-лучевая асимметричная звезда магов.
Переход от 2D к 3D: каждый луч — отдельное измерение отношения.
Деформация лучей = семантические веса измерений.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import math


N_RAYS = 7

# Семантические метки 7 лучей
RAY_LABELS = [
    "strength",    # 0 — сила связи
    "direction",   # 1 — направленность (однонаправл./взаимная)
    "temporal",    # 2 — временная динамика
    "confidence",  # 3 — уверенность/достоверность
    "scale",       # 4 — масштаб (микро ↔ макро)
    "context",     # 5 — контекстуальная близость
    "source",      # 6 — качество источника
]


@dataclass
class Ray:
    """Один луч семилучевой звезды."""
    index:  int      # 0–6
    label:  str      # семантическая метка
    length: float    # вес/интенсивность [0, 1]
    angle:  float    # угол в 3D-пространстве (радианы)
    curve:  float    # кривизна луча: 0=прямой, 1=максимально изогнутый
    z:      float    # высота кончика луча в 3D (выход из плоскости)

    @property
    def endpoint_3d(self) -> tuple[float, float, float]:
        """Координата кончика луча в 3D."""
        x = self.length * math.cos(self.angle)
        y = self.length * math.sin(self.angle)
        return (x, y, self.z)

    @property
    def endpoint_2d(self) -> tuple[float, float]:
        return (self.length * math.cos(self.angle),
                self.length * math.sin(self.angle))


@dataclass
class HeptagramSignature:
    """
    7-лучевая звезда с деформированными лучами.
    Описывает многомерное отношение между N сущностями.
    """
    rays:        list[Ray]   # 7 лучей, каждый разного размера
    center:      tuple[float, float, float] = (0.0, 0.0, 0.0)
    is_3d:       bool = True             # True = лучи выходят в 3D
    total_energy: float = 0.0           # суммарная сила отношения

    def __post_init__(self):
        self.total_energy = sum(r.length for r in self.rays) / N_RAYS

    @property
    def dominant_ray(self) -> Ray:
        """Луч с наибольшим весом — главное измерение отношения."""
        return max(self.rays, key=lambda r: r.length)

    @property
    def symmetry_score(self) -> float:
        """0 = идеально асимметрична, 1 = идеально симметрична."""
        lengths = [r.length for r in self.rays]
        mean    = sum(lengths) / len(lengths)
        variance = sum((l - mean)**2 for l in lengths) / len(lengths)
        return 1.0 / (1.0 + variance * 10)

    def to_vector(self) -> list[float]:
        """Вектор для сравнения звёзд: [length_0..6, curve_0..6, z_0..6]."""
        return ([r.length for r in self.rays] +
                [r.curve  for r in self.rays] +
                [r.z      for r in self.rays])


def heptagram_distance(a: HeptagramSignature, b: HeptagramSignature) -> float:
    """Расстояние между двумя звёздами (L2 по вектору)."""
    va, vb = a.to_vector(), b.to_vector()
    return math.sqrt(sum((x-y)**2 for x, y in zip(va, vb)))


def build_heptagram_signature(
    relation_weights: dict[str, float]
) -> HeptagramSignature:
    """
    Построить HeptagramSignature из словаря весов по 7 измерениям.

    relation_weights: {
        "strength":   0.8,
        "direction":  0.3,
        "temporal":   0.5,
        "confidence": 0.9,
        "scale":      0.4,
        "context":    0.7,
        "source":     0.6,
    }
    """
    base_angle = 2 * math.pi / N_RAYS
    rays = []
    for i, label in enumerate(RAY_LABELS):
        length = max(0.0, min(1.0, relation_weights.get(label, 0.5)))
        angle  = base_angle * i
        # деформация: более длинные лучи слегка закручиваются
        curve  = length * 0.3
        # 3D высота: длинные лучи поднимаются из плоскости
        z      = length * 0.5 * math.sin(angle)
        rays.append(Ray(index=i, label=label, length=length,
                        angle=angle, curve=curve, z=z))
    return HeptagramSignature(rays=rays, is_3d=True)


def heptagram_from_edge_weights(
    nodes: list[str],
    edge_weights: list[tuple[str, str, float]],
) -> HeptagramSignature:
    """
    Автоматически вычислить HeptagramSignature из набора рёбер кластера.
    Метрики извлекаются из статистики рёбер.
    """
    if not edge_weights:
        return build_heptagram_signature({})

    weights = [w for _, _, w in edge_weights]
    n_edges = len(weights)
    mean_w  = sum(weights) / n_edges
    max_w   = max(weights)
    min_w   = min(weights)
    variance = sum((w - mean_w)**2 for w in weights) / n_edges

    # направленность: доля несимметричных рёбер
    pairs = {(a,b): w for a,b,w in edge_weights}
    directed = sum(1 for a,b,w in edge_weights
                   if (b,a) not in pairs or abs(pairs[(b,a)]-w) > 0.1)
    direction = directed / n_edges

    return build_heptagram_signature({
        "strength":   mean_w,
        "direction":  direction,
        "temporal":   variance,               # нестабильность ≈ временная динамика
        "confidence": 1.0 - (max_w - min_w), # разброс ≈ неопределённость
        "scale":      min(1.0, n_edges / 10), # масштаб по числу рёбер
        "context":    max_w,
        "source":     mean_w * (1.0 - variance),
    })
