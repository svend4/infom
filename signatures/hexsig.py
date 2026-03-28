"""
HexSignature — Q6-позиция сущности в 6-мерном гиперкубе.
Адаптировано из meta/libs/hexcore/hexcore.py
"""
from __future__ import annotations
from dataclasses import dataclass
import math


# ── Q6 константы ────────────────────────────────────────────────────────────

N_DIMS    = 6     # размерность
N_NODES   = 64    # 2^6 вершин
MAX_DIST  = 6     # диаметр графа
DEGREE    = 6     # каждая вершина имеет 6 соседей

# Спектр расстояний от любой вершины: [1, 6, 15, 20, 15, 6, 1]
DISTANCE_SPECTRUM = [math.comb(N_DIMS, r) for r in range(N_DIMS + 1)]


@dataclass
class HexSignature:
    """Q6-позиция: кластер как вершина 6-мерного гиперкуба."""
    hex_id:     int            # 0-63 — позиция в Q6
    voronoi_id: int            # к какому центру Voronoi принадлежит
    hamming_r:  int            # радиус занимаемого Hamming-шара
    bits:       tuple[int,...] # 6-битный вектор (b5, b4, b3, b2, b1, b0)

    @property
    def archetype_bits(self) -> dict[str, int]:
        """Семантические оси Q6 (как в pseudorag)."""
        b = self.bits
        return {
            "material":  b[0],   # 0=абстрактное, 1=материальное
            "dynamic":   b[1],   # 0=статичное,   1=динамичное
            "complex":   b[2],   # 0=элементарное, 1=сложное
            "ordered":   b[3],   # 0=флюидное,    1=упорядоченное
            "local":     b[4],   # 0=глобальное,  1=локальное
            "explicit":  b[5],   # 0=неявное,     1=явное
        }


# ── базовые операции Q6 ──────────────────────────────────────────────────────

def to_bits(h: int) -> tuple[int,...]:
    return tuple((h >> i) & 1 for i in range(N_DIMS))


def from_bits(bits: tuple[int,...]) -> int:
    return sum(b << i for i, b in enumerate(bits))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count('1')


def neighbors(h: int) -> list[int]:
    return [h ^ (1 << i) for i in range(N_DIMS)]


def antipode(h: int) -> int:
    return h ^ (N_NODES - 1)


def hamming_ball(center: int, radius: int) -> list[int]:
    return [h for h in range(N_NODES) if hamming(center, h) <= radius]


def hamming_sphere(center: int, radius: int) -> list[int]:
    return [h for h in range(N_NODES) if hamming(center, h) == radius]


def bfs_distances(start: int) -> list[int]:
    dist = [-1] * N_NODES
    dist[start] = 0
    queue = [start]
    for h in queue:
        for nb in neighbors(h):
            if dist[nb] < 0:
                dist[nb] = dist[h] + 1
                queue.append(nb)
    return dist


def shortest_path(a: int, b: int) -> list[int]:
    """BFS кратчайший путь от a до b в Q6."""
    if a == b:
        return [a]
    dist = bfs_distances(a)
    path = [b]
    cur  = b
    while cur != a:
        for nb in neighbors(cur):
            if dist[nb] == dist[cur] - 1:
                path.append(nb)
                cur = nb
                break
    path.reverse()
    return path


def metric_interval(u: int, v: int) -> list[int]:
    """Все вершины на кратчайшем пути между u и v."""
    d_uv = hamming(u, v)
    du   = bfs_distances(u)
    dv   = bfs_distances(v)
    return [h for h in range(N_NODES) if du[h] + dv[h] == d_uv]


def median(points: list[int]) -> int:
    """Медиана (Steiner point) — побитовое большинство."""
    result = 0
    for bit in range(N_DIMS):
        ones = sum((p >> bit) & 1 for p in points)
        if ones > len(points) / 2:
            result |= (1 << bit)
    return result


# ── Voronoi на Q6 ────────────────────────────────────────────────────────────

def voronoi_cells(centers: list[int]) -> dict[int, list[int]]:
    """Разбивка Q6 на ячейки Voronoi по набору центров."""
    cells: dict[int, list[int]] = {c: [] for c in centers}
    for h in range(N_NODES):
        nearest = min(centers, key=lambda c: hamming(h, c))
        cells[nearest].append(h)
    return cells


def delaunay_graph(centers: list[int]) -> list[tuple[int, int]]:
    """Граф Делоне: два центра связаны если их ячейки Voronoi граничат."""
    cells = voronoi_cells(centers)
    edges = set()
    for c1 in centers:
        for node in cells[c1]:
            for nb in neighbors(node):
                for c2 in centers:
                    if c2 != c1 and nb in cells[c2]:
                        edge = (min(c1,c2), max(c1,c2))
                        edges.add(edge)
    return list(edges)


# ── Sphere packing ────────────────────────────────────────────────────────────

def packing_number(radius: int) -> list[int]:
    """Жадная упаковка шарами радиуса r — непересекающиеся шестиугольные ячейки."""
    covered = set()
    centers = []
    for h in range(N_NODES):
        if h not in covered:
            centers.append(h)
            covered.update(hamming_ball(h, radius))
    return centers


def is_perfect_code(centers: list[int], radius: int) -> bool:
    """Идеальная упаковка: шары не пересекаются и покрывают весь Q6."""
    all_nodes = set()
    for c in centers:
        ball = set(hamming_ball(c, radius))
        if ball & all_nodes:
            return False
        all_nodes |= ball
    return len(all_nodes) == N_NODES


# ── embedding → Q6 проекция ─────────────────────────────────────────────────

def embed_to_q6(embedding: list[float]) -> int:
    """
    Проецирует непрерывный embedding любой размерности в Q6 (6 бит, 64 ячейки).

    Алгоритм проекции N→6:
      N <= 6  : padding нулями до 6D (как раньше)
      N > 6   : average-pool — делим вектор на 6 равных чанков, берём среднее каждого.
                Это сохраняет информацию со всей размерности, а не только первых 6 элементов.
                Для N=32: чанки по ~5 элементов → лучшее разрешение, меньше коллизий.
                Для N=768 (bge-m3): чанки по ~128 элементов → полное использование.

    Бинаризация: бит=1 если значение ВЫШЕ среднего по 6D-проекции.
    """
    n = len(embedding)
    if n <= N_DIMS:
        vals = [embedding[i] if i < n else 0.0 for i in range(N_DIMS)]
    else:
        # Average-pool: N dims → 6 dims
        chunk = n / N_DIMS
        vals = []
        for d in range(N_DIMS):
            start = int(d * chunk)
            end   = int((d + 1) * chunk)
            if end <= start:
                end = start + 1
            chunk_vals = embedding[start:end]
            vals.append(sum(chunk_vals) / len(chunk_vals))

    mean = sum(vals) / N_DIMS
    if all(abs(v - mean) < 1e-10 for v in vals):
        bits = [1 if v > 0.5 else 0 for v in vals]
    else:
        bits = [1 if v > mean else 0 for v in vals]
    return from_bits(tuple(bits))


def build_hex_signature(
    embedding: list[float],
    voronoi_centers: list[int] | None = None
) -> HexSignature:
    hex_id = embed_to_q6(embedding)
    bits   = to_bits(hex_id)

    if voronoi_centers:
        vid = min(voronoi_centers, key=lambda c: hamming(hex_id, c))
        r   = hamming(hex_id, vid)
    else:
        vid = hex_id
        r   = 0

    return HexSignature(hex_id=hex_id, voronoi_id=vid, hamming_r=r, bits=bits)
