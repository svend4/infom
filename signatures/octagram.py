"""
OctagramSignature — 8-лучевая звезда / роза ветров / звезда НАТО.
Задаёт 3D-компас для ориентации кластера в пространстве знаний.
Используется для прозрачного скелета: небоскрёб, ракушка, патефон.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import math


N_RAYS = 8

# 8 направлений как компас
COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]

# Семантика осей (пары противоположных направлений)
AXIS_SEMANTICS = {
    ("N",  "S"):  ("abstract",  "concrete"),    # вертикаль абстракции
    ("E",  "W"):  ("future",    "past"),         # ось времени
    ("NE", "SW"): ("complex",   "simple"),       # ось сложности
    ("NW", "SE"): ("global",    "local"),        # ось масштаба
}


class SkeletonType(Enum):
    """Тип 3D каркаса который строит эта звезда."""
    TOWER    = "tower"     # небоскрёб — иерархичный
    SHELL    = "shell"     # ракушка — спиральный
    CONE     = "cone"      # патефон — расширяющийся
    FRACTAL  = "fractal"   # фрактальный каркас
    SPHERE   = "sphere"    # равномерный (все лучи одинаковые)


@dataclass
class OctaRay:
    """Один луч восьмиконечной звезды."""
    direction: str          # "N", "NE", "E"...
    length:    float        # 0..1
    angle_2d:  float        # угол в плоскости XY
    elevation: float        # угол к горизонту (3D: -π/2 .. π/2)
    weight:    float        # семантический вес направления

    @property
    def endpoint_3d(self) -> tuple[float, float, float]:
        r = self.length
        x = r * math.cos(self.elevation) * math.cos(self.angle_2d)
        y = r * math.cos(self.elevation) * math.sin(self.angle_2d)
        z = r * math.sin(self.elevation)
        return (x, y, z)


@dataclass
class OctagramSignature:
    """
    8-лучевая звезда — 3D компас кластера знаний.
    Прозрачный скелет: позволяет «видеть сквозь» к нижним уровням.
    """
    rays:         list[OctaRay]
    skeleton_type: SkeletonType
    center_3d:    tuple[float, float, float] = (0.0, 0.0, 0.0)

    @property
    def dominant_axis(self) -> tuple[str, str]:
        """Главная ось — пара противоположных лучей с максимальной суммой."""
        compass_map = {r.direction: r.length for r in self.rays}
        best, best_sum = ("N","S"), 0.0
        for (d1, d2), _ in AXIS_SEMANTICS.items():
            s = compass_map.get(d1, 0) + compass_map.get(d2, 0)
            if s > best_sum:
                best, best_sum = (d1, d2), s
        return best

    @property
    def is_flat(self) -> bool:
        """True если все лучи лежат в одной плоскости (elevation ≈ 0)."""
        return all(abs(r.elevation) < 0.1 for r in self.rays)

    def to_vector(self) -> list[float]:
        return [r.length for r in self.rays] + [r.elevation for r in self.rays]

    def skeleton_vertices(self) -> list[tuple[float,float,float]]:
        """Вершины 3D скелета (кончики лучей + центр)."""
        verts = [self.center_3d]
        verts += [r.endpoint_3d for r in self.rays]
        return verts

    def skeleton_edges(self) -> list[tuple[int,int]]:
        """Рёбра 3D скелета: центр(0) → каждый луч(1..8) + периметр."""
        edges = [(0, i+1) for i in range(N_RAYS)]
        for i in range(N_RAYS):
            edges.append((i+1, (i+1)%N_RAYS + 1))
        return edges


def octagram_distance(a: OctagramSignature, b: OctagramSignature) -> float:
    va, vb = a.to_vector(), b.to_vector()
    return math.sqrt(sum((x-y)**2 for x, y in zip(va, vb)))


def _detect_skeleton_type(rays: list[OctaRay]) -> SkeletonType:
    lengths  = [r.length for r in rays]
    elevs    = [abs(r.elevation) for r in rays]
    mean_l   = sum(lengths) / len(lengths)
    variance = sum((l-mean_l)**2 for l in lengths) / len(lengths)
    mean_e   = sum(elevs) / len(elevs)

    if variance < 0.02:
        return SkeletonType.SPHERE
    if mean_e > 0.4:
        spiral = sum(1 for i in range(len(rays))
                     if rays[i].elevation > rays[i-1].elevation)
        return SkeletonType.SHELL if spiral > 4 else SkeletonType.CONE
    if lengths[0] > lengths[4] * 1.5:  # N намного больше S
        return SkeletonType.TOWER
    return SkeletonType.FRACTAL


def build_octagram_signature(
    direction_weights: dict[str, float],
    elevation_profile: dict[str, float] | None = None,
) -> OctagramSignature:
    """
    Построить OctagramSignature из весов по 8 направлениям.

    direction_weights: {"N": 0.8, "NE": 0.3, "E": 0.5, ...}
    elevation_profile: {"N": 0.2, "NE": 0.1, ...}  (опционально, высота луча)
    """
    if elevation_profile is None:
        elevation_profile = {}

    base_angle = 2 * math.pi / N_RAYS
    rays = []
    for i, compass_dir in enumerate(COMPASS):
        length    = max(0.0, min(1.0, direction_weights.get(compass_dir, 0.5)))
        angle_2d  = base_angle * i
        elevation = elevation_profile.get(compass_dir, 0.0)
        # автоматически поднимаем «тяжёлые» лучи в 3D
        if elevation == 0.0:
            elevation = (length - 0.5) * math.pi / 4
        rays.append(OctaRay(
            direction = compass_dir,
            length    = length,
            angle_2d  = angle_2d,
            elevation = elevation,
            weight    = length,
        ))

    skeleton = _detect_skeleton_type(rays)
    return OctagramSignature(rays=rays, skeleton_type=skeleton)


def build_shell_octagram(n_turns: float = 1.5) -> OctagramSignature:
    """
    Ракушка Фибоначчи — лучи закручиваются по логарифмической спирали.
    Используется для эволюции понятий: от простого к сложному.
    """
    phi  = (1 + math.sqrt(5)) / 2   # золотое сечение
    rays = []
    for i, compass_dir in enumerate(COMPASS):
        t      = i / N_RAYS
        length = 0.3 + 0.7 * (phi ** (t * n_turns)) / (phi ** n_turns)
        angle_2d  = 2 * math.pi * t
        elevation = math.pi / 4 * math.sin(2 * math.pi * t)
        rays.append(OctaRay(
            direction = compass_dir,
            length    = min(1.0, length),
            angle_2d  = angle_2d,
            elevation = elevation,
            weight    = length,
        ))
    return OctagramSignature(rays=rays, skeleton_type=SkeletonType.SHELL)


def build_tower_octagram(n_levels: int = 4) -> OctagramSignature:
    """
    Небоскрёб — лучи на разных уровнях иерархии.
    N=верхний уровень абстракции, S=нижний конкретный.
    """
    level_heights = [i / (n_levels - 1) for i in range(n_levels)]
    directions    = dict(zip(COMPASS, [0.9, 0.6, 0.7, 0.5, 0.8, 0.4, 0.6, 0.5]))
    elevations    = {
        "N":  math.pi/3,   # высоко — абстрактное
        "NE": math.pi/6,
        "E":  0.0,
        "SE": -math.pi/6,
        "S":  -math.pi/3,  # низко — конкретное
        "SW": -math.pi/4,
        "W":  0.0,
        "NW": math.pi/4,
    }
    return build_octagram_signature(directions, elevations)
