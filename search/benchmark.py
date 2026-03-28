"""
Бенчмарк Multi-Projection Q6 LSH.

Измеряет recall и покрытие для разного числа проекций:
  recall = доля правильных ближайших соседей, попавших в union Hamming ball

Теоретически при независимых проекциях:
  recall(k) = 1 - (1 - recall_1)^k

На практике проекции коррелируют → recall(k) ниже теории, но всё равно растёт.
"""
from __future__ import annotations
import math

from search.multi_lsh import MultiProjectionQ6
from signatures import hamming, embed_to_q6


def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na  = math.sqrt(sum(x*x for x in a[:n])) or 1.0
    nb  = math.sqrt(sum(x*x for x in b[:n])) or 1.0
    return dot / (na * nb)


def _lcg_vec(seed: int, dim: int = 6) -> list[float]:
    """Детерминированный псевдо-эмбеддинг."""
    vec = []
    for i in range(dim):
        seed = (seed * 1664525 + 1013904223) & 0xFFFFFFFF
        vec.append((seed / 0x7FFFFFFF) - 1.0)
    return vec


def run_recall_benchmark(
    n_vectors:     int = 500,
    radius:        int = 1,
    top_k:         int = 10,
    n_projections: list[int] = (1, 2, 3, 5),
    seed:          int = 42,
) -> list[dict]:
    """
    Генерируем n_vectors случайных 6D векторов.
    Для каждого query находим top_k ближайших соседей по cosine.
    Проверяем: попадают ли они в union Hamming ball при k проекциях.

    Возвращает список dict с полями:
      n_proj, recall, avg_ball_size, avg_candidates
    """
    # генерируем векторы
    vectors = [_lcg_vec(seed + i * 31) for i in range(n_vectors)]

    results = []
    for n_proj in n_projections:
        mlsh = MultiProjectionQ6(n_projections=n_proj, seed=seed)

        total_hits = 0
        total_possible = 0
        total_ball_size = 0.0
        n_queries = min(100, n_vectors)

        for qi in range(n_queries):
            query = vectors[qi]

            # true nearest neighbors (cosine)
            sims = sorted(
                [(i, _cosine(query, vectors[i]))
                 for i in range(n_vectors) if i != qi],
                key=lambda x: -x[1],
            )
            true_top = {idx for idx, _ in sims[:top_k]}

            # union Hamming ball
            ball = mlsh.union_hamming_ball(query, radius)
            total_ball_size += len(ball)

            # какие из true_top попали в ball (по их hex_id)
            for idx in true_top:
                hex_id = embed_to_q6(vectors[idx])
                if hex_id in ball:
                    total_hits += 1
            total_possible += len(true_top)

        recall = total_hits / total_possible if total_possible > 0 else 0.0
        avg_ball = total_ball_size / n_queries

        results.append({
            "n_proj":       n_proj,
            "recall":       recall,
            "avg_ball_size": avg_ball,
            "coverage_pct": avg_ball / 64 * 100,
        })

    return results


def format_benchmark(results: list[dict]) -> str:
    lines = [
        "Multi-Projection Q6 LSH — Recall Benchmark",
        f"{'Проекций':>10}  {'Recall':>8}  {'Ball/64':>8}  {'Coverage':>10}",
        "─" * 46,
    ]
    for r in results:
        bar = "█" * int(r["recall"] * 20) + "░" * (20 - int(r["recall"] * 20))
        lines.append(
            f"{r['n_proj']:>10}  "
            f"{r['recall']*100:>7.1f}%  "
            f"{r['avg_ball_size']:>6.1f}/64  "
            f"[{bar}]"
        )
    # теоретические значения
    lines.append("")
    lines.append("Теория (независимые проекции):")
    if results:
        recall_1 = results[0]["recall"]
        for r in results:
            theory = 1 - (1 - recall_1) ** r["n_proj"]
            lines.append(f"  k={r['n_proj']}: {theory*100:.1f}%  (эмпирически: {r['recall']*100:.1f}%)")
    return "\n".join(lines)
