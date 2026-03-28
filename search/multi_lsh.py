"""
MultiProjectionQ6 — несколько независимых Q6 проекций для улучшенного recall.

Идея из Habr https://habr.com/ru/articles/1011992/:
  Один LSH индекс даёт ложные промахи: релевантные объекты могут оказаться
  за пределами Hamming ball из-за случайного направления проекции.
  Решение: k независимых случайных Q6 проекций → объединение Hamming ball.

Алгоритм:
  1. Генерируем k случайных ортогональных матриц 6×6 (схема QR-разложения)
  2. Для каждого вектора: проецируем через каждую матрицу → получаем k hex_id
  3. Stage 1 HNSW: берём объединение Hamming ball от всех k hex_id
  4. Результат: recall растёт с O(1-p) до O((1-p)^k) где p=вероятность промаха

Примерные цифры:
  1 проекция, radius=1: recall ~80%
  3 проекции, radius=1: recall ~99.2%  (1 - 0.2^3)
  Но кандидатов этапа1 тоже больше → важен быстрый этап 2
"""
from __future__ import annotations
import math

from signatures import embed_to_q6, hamming_ball, hamming


# ── утилиты линейной алгебры (stdlib only) ───────────────────────────────────

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]

def _sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]

def _norm(a: list[float]) -> float:
    return math.sqrt(_dot(a, a))

def _normalize(a: list[float]) -> list[float]:
    n = _norm(a) or 1.0
    return _scale(a, 1.0 / n)

def _lcg_float(seed: int) -> tuple[float, int]:
    """Линейный конгруэнтный генератор → float в (-1, 1)."""
    seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
    return (seed / 0x7FFFFFFF) - 1.0, seed


def _random_orthogonal(n: int, seed: int) -> list[list[float]]:
    """
    Генерируем случайную ортогональную матрицу n×n методом Gram-Schmidt.
    Детерминировано по seed.
    """
    vecs: list[list[float]] = []
    for i in range(n):
        # случайный вектор
        v = []
        for j in range(n):
            val, seed = _lcg_float(seed * 31 + i * 97 + j * 7 + 13)
            v.append(val)
        # ортогонализация Gram-Schmidt
        for u in vecs:
            proj = _scale(u, _dot(v, u))
            v = _sub(v, proj)
        nv = _norm(v)
        if nv < 1e-10:
            # дополняем единичным вектором если коллинеарен
            v = [1.0 if j == i else 0.0 for j in range(n)]
        else:
            v = _scale(v, 1.0 / nv)
        vecs.append(v)
    return vecs


def _matvec(mat: list[list[float]], v: list[float]) -> list[float]:
    """Умножение матрицы на вектор."""
    return [_dot(row, v) for row in mat]


# ── основной класс ────────────────────────────────────────────────────────────

class MultiProjectionQ6:
    """
    k независимых Q6 проекций для улучшенного recall в Stage 1 HNSW.

    n_projections: количество проекций (3-5 оптимально)
    seed:          начальное значение для детерминированной генерации матриц
    """

    def __init__(self, n_projections: int = 3, seed: int = 42):
        self.n_projections = n_projections
        # генерируем k ортогональных матриц 6×6
        self._matrices: list[list[list[float]]] = [
            _random_orthogonal(6, seed + i * 1000)
            for i in range(n_projections)
        ]

    def project(self, embedding: list[float], idx: int) -> list[float]:
        """Проецируем embedding через матрицу idx."""
        v = [embedding[i] if i < len(embedding) else 0.0 for i in range(6)]
        return _matvec(self._matrices[idx], v)

    def hash(self, embedding: list[float], idx: int) -> int:
        """Q6 hex_id для проекции idx."""
        projected = self.project(embedding, idx)
        return embed_to_q6(projected)

    def hash_all(self, embedding: list[float]) -> list[int]:
        """k Q6 hex_id — по одному на каждую проекцию."""
        return [self.hash(embedding, i) for i in range(self.n_projections)]

    def union_hamming_ball(
        self, embedding: list[float], radius: int
    ) -> set[int]:
        """
        Объединение Hamming ball от всех k проекций.

        Recall растёт экспоненциально:
          recall(k проекций) ≈ 1 - (1 - recall_1)^k
        """
        result: set[int] = set()
        for i in range(self.n_projections):
            hex_id = self.hash(embedding, i)
            result.update(hamming_ball(hex_id, radius))
        return result

    def distances(
        self, embedding: list[float], other_embedding: list[float]
    ) -> list[int]:
        """Hamming расстояния между embedding и other_embedding для каждой проекции."""
        return [
            hamming(self.hash(embedding, i), self.hash(other_embedding, i))
            for i in range(self.n_projections)
        ]

    def min_distance(
        self, embedding: list[float], other_embedding: list[float]
    ) -> int:
        """Минимальное Hamming расстояние по всем проекциям (лучшая оценка близости)."""
        return min(self.distances(embedding, other_embedding))

    def avg_distance(
        self, embedding: list[float], other_embedding: list[float]
    ) -> float:
        """Среднее Hamming расстояние — устойчивая оценка близости."""
        dists = self.distances(embedding, other_embedding)
        return sum(dists) / len(dists)

    def coverage_score(
        self, embedding: list[float], hex_id: int, radius: int
    ) -> float:
        """
        Доля проекций, в чьих Hamming ball находится hex_id.
        Чем больше — тем достовернее кандидат.
        """
        count = 0
        for i in range(self.n_projections):
            my_hex = self.hash(embedding, i)
            if hamming(my_hex, hex_id) <= radius:
                count += 1
        return count / self.n_projections

    # ── информация ──────────────────────────────────────────────────────────

    def info(self) -> str:
        lines = [
            f"MultiProjectionQ6: {self.n_projections} проекций",
            f"  Ожидаемый recall при radius=1: "
            f"≈{100*(1 - 0.2**self.n_projections):.1f}%",
            f"  Размер union Hamming ball (r=1): "
            f"~{min(64, 7 * self.n_projections)} из 64",
        ]
        return "\n".join(lines)
