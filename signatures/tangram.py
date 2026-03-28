"""
TangramSignature — геометрическое описание внутренней формы кластера.
Адаптировано из meta2/puzzle_reconstruction/algorithms/tangram/
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
import math


class ShapeClass(Enum):
    TRIANGLE   = "triangle"    # 3 ноды
    RECTANGLE  = "rectangle"   # 4 ноды
    TRAPEZOID  = "trapezoid"   # 4 ноды (неравн.)
    PENTAGON   = "pentagon"    # 5 нод
    HEXAGON    = "hexagon"     # 6 нод → сота
    HEPTAGON   = "heptagon"    # 7 нод → звезда магов
    OCTAGON    = "octagon"     # 8 нод → роза ветров
    POLYGON    = "polygon"     # N нод


@dataclass
class TangramSignature:
    """Геометрическое описание внутреннего полигона кластера нод."""
    polygon:     list[tuple[float, float]]  # нормализованные вершины
    shape_class: ShapeClass
    centroid:    tuple[float, float]
    angle:       float   # угол главной оси, радианы
    scale:       float   # диагональ описанного прямоугольника = 1
    area:        float   # площадь нормализованного полигона

    @property
    def n_vertices(self) -> int:
        return len(self.polygon)


# ── утилиты ────────────────────────────────────────────────────────────────

def compute_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    n = len(points)
    return (sum(p[0] for p in points) / n,
            sum(p[1] for p in points) / n)


def compute_area(points: list[tuple[float, float]]) -> float:
    """Формула Гаусса (Shoelace)."""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def compute_interior_angles(points: list[tuple[float, float]]) -> list[float]:
    n = len(points)
    angles = []
    for i in range(n):
        prev = points[(i - 1) % n]
        curr = points[i]
        nxt  = points[(i + 1) % n]
        v1 = (prev[0] - curr[0], prev[1] - curr[1])
        v2 = (nxt[0]  - curr[0], nxt[1]  - curr[1])
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        n1  = math.hypot(*v1)
        n2  = math.hypot(*v2)
        if n1 < 1e-10 or n2 < 1e-10:
            angles.append(0.0)
        else:
            cos_a = max(-1.0, min(1.0, dot / (n1 * n2)))
            angles.append(math.acos(cos_a))
    return angles


def _classify_quad(points: list[tuple[float, float]]) -> ShapeClass:
    angles = compute_interior_angles(points)
    right  = sum(1 for a in angles if abs(a - math.pi/2) < math.radians(15))
    if right == 4:
        return ShapeClass.RECTANGLE

    def cross_ratio(p1, p2, p3, p4):
        dx1, dy1 = p2[0]-p1[0], p2[1]-p1[1]
        dx2, dy2 = p4[0]-p3[0], p4[1]-p3[1]
        cross = abs(dx1*dy2 - dy1*dx2)
        dot   = abs(dx1*dx2 + dy1*dy2)
        return cross / (dot + 1e-10)

    if cross_ratio(points[0], points[1], points[2], points[3]) < 0.10:
        return ShapeClass.RECTANGLE   # параллелограмм → трактуем как прямоугольник

    for i in range(2):
        p1, p2 = points[i], points[(i+1)%4]
        p3, p4 = points[(i+2)%4], points[(i+3)%4]
        if cross_ratio(p1, p2, p3, p4) < 0.15:
            return ShapeClass.TRAPEZOID

    return ShapeClass.POLYGON


def classify_shape(points: list[tuple[float, float]]) -> ShapeClass:
    n = len(points)
    if n <= 2:  return ShapeClass.POLYGON
    if n == 3:  return ShapeClass.TRIANGLE
    if n == 4:  return _classify_quad(points)
    if n == 5:  return ShapeClass.PENTAGON
    if n == 6:  return ShapeClass.HEXAGON
    if n == 7:  return ShapeClass.HEPTAGON
    if n == 8:  return ShapeClass.OCTAGON
    return ShapeClass.POLYGON


def normalize_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Нормализация: центр в (0,0), диагональ = 1."""
    cx, cy = compute_centroid(points)
    shifted = [(p[0] - cx, p[1] - cy) for p in points]
    max_d = max(math.hypot(*p) for p in shifted) or 1.0
    return [(p[0]/max_d, p[1]/max_d) for p in shifted]


def compute_main_axis_angle(points: list[tuple[float, float]]) -> float:
    """Угол главной оси через ковариационную матрицу."""
    cx, cy = compute_centroid(points)
    xx = sum((p[0]-cx)**2 for p in points)
    xy = sum((p[0]-cx)*(p[1]-cy) for p in points)
    return math.atan2(2*xy, xx) / 2.0


def build_tangram_signature(
    node_positions: list[tuple[float, float]]
) -> TangramSignature:
    """Построить TangramSignature из позиций нод кластера."""
    hull = convex_hull(node_positions)
    norm = normalize_polygon(hull)
    return TangramSignature(
        polygon     = norm,
        shape_class = classify_shape(norm),
        centroid    = compute_centroid(norm),
        angle       = compute_main_axis_angle(norm),
        scale       = 1.0,
        area        = compute_area(norm),
    )


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Быстрый алгоритм выпуклой оболочки (Gift Wrapping)."""
    if len(points) <= 2:
        return points

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    pts = sorted(set(points))
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]
