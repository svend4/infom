"""
FractalSignature — фрактальное описание границы кластера.
Адаптировано из meta2/puzzle_reconstruction/algorithms/fractal/
"""
from __future__ import annotations
from dataclasses import dataclass
import math


@dataclass
class FractalSignature:
    """Фрактальное описание одной границы между сообществами."""
    fd_box:     float        # фракт. размерность (Box-counting): 1.0=прямая, 2.0=плоскость
    fd_divider: float        # фракт. размерность (Divider/Richardson)
    ifs_coeffs: list[float]  # коэффициенты IFS (Barnsley) — компактное описание кривой
    css_image:  list[tuple[float, list[float]]]  # Curvature Scale Space [(sigma, zero_crossings)]
    chain_code: str          # цепной код Фримана (8-направлений) — хэш формы
    curve:      list[tuple[float, float]]        # параметрическая кривая (N, 2)


# ── Box-counting ────────────────────────────────────────────────────────────

def box_counting_dimension(curve: list[tuple[float, float]],
                           n_scales: int = 8) -> float:
    """
    Фрактальная размерность методом ячеек.
    Считаем сколько ячеек размера ε покрывает кривую при разных ε.
    """
    if len(curve) < 2:
        return 1.0

    xs = [p[0] for p in curve]
    ys = [p[1] for p in curve]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    span = max(x_max - x_min, y_max - y_min, 1e-10)

    counts, epsilons = [], []
    for k in range(1, n_scales + 1):
        eps = span / (2 ** k)
        boxes = set()
        for p in curve:
            bx = int((p[0] - x_min) / eps)
            by = int((p[1] - y_min) / eps)
            boxes.add((bx, by))
        if boxes:
            counts.append(math.log(len(boxes)))
            epsilons.append(math.log(1.0 / eps))

    if len(counts) < 2:
        return 1.0
    return _linear_regression_slope(epsilons, counts)


def _linear_regression_slope(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 1.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num   = sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    denom = sum((xs[i]-mx)**2         for i in range(n))
    return num / (denom + 1e-10)


# ── Divider / Richardson ────────────────────────────────────────────────────

def divider_dimension(curve: list[tuple[float, float]],
                      n_steps: int = 8) -> float:
    """Фрактальная размерность методом циркуля (Richardson)."""
    if len(curve) < 2:
        return 1.0

    span = max(
        max(p[0] for p in curve) - min(p[0] for p in curve),
        max(p[1] for p in curve) - min(p[1] for p in curve),
        1e-10
    )

    counts, epsilons = [], []
    for k in range(1, n_steps + 1):
        step = span / (2 ** k)
        count = _count_divider_steps(curve, step)
        if count > 0:
            counts.append(math.log(count))
            epsilons.append(math.log(1.0 / step))

    if len(counts) < 2:
        return 1.0
    return _linear_regression_slope(epsilons, counts)


def _count_divider_steps(curve: list[tuple[float, float]], step: float) -> int:
    count  = 0
    pos    = 0
    cx, cy = curve[0]
    while pos < len(curve) - 1:
        for j in range(pos + 1, len(curve)):
            dx = curve[j][0] - cx
            dy = curve[j][1] - cy
            if math.hypot(dx, dy) >= step:
                cx, cy = curve[j]
                pos = j
                count += 1
                break
        else:
            break
    return max(count, 1)


# ── IFS (Iterated Function System / Barnsley) ───────────────────────────────

def fit_ifs_coefficients(curve: list[tuple[float, float]],
                         n_transforms: int = 8) -> list[float]:
    """
    Аппроксимация кривой коэффициентами IFS.
    Возвращает вертикальные коэффициенты сжатия d_k.
    """
    if len(curve) < 2:
        return [0.0] * n_transforms

    profile = _extract_height_profile(curve)
    n = len(profile)
    seg_size = max(1, n // n_transforms)
    coeffs = []

    for k in range(n_transforms):
        start = k * seg_size
        end   = min(start + seg_size, n)
        seg   = profile[start:end]
        if len(seg) < 2:
            coeffs.append(0.0)
            continue
        full_resampled = _resample_1d(profile, len(seg))
        num   = sum(seg[i] * full_resampled[i] for i in range(len(seg)))
        denom = sum(x**2 for x in full_resampled) + 1e-10
        d_k   = max(-0.95, min(0.95, num / denom))
        coeffs.append(d_k)

    return coeffs


def ifs_distance(a: list[float], b: list[float]) -> float:
    """L2-норма разности коэффициентов IFS — быстрый скрининг похожести."""
    n = min(len(a), len(b))
    return math.sqrt(sum((a[i]-b[i])**2 for i in range(n)))


def _extract_height_profile(curve: list[tuple[float, float]]) -> list[float]:
    p0, p1 = curve[0], curve[-1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy/length, dx/length
    return [nx*(p[0]-p0[0]) + ny*(p[1]-p0[1]) for p in curve]


def _resample_1d(arr: list[float], target: int) -> list[float]:
    if len(arr) == target:
        return arr[:]
    result = []
    for i in range(target):
        t   = i / max(target - 1, 1) * (len(arr) - 1)
        lo  = int(t)
        hi  = min(lo + 1, len(arr) - 1)
        frac = t - lo
        result.append(arr[lo] * (1-frac) + arr[hi] * frac)
    return result


# ── Curvature Scale Space (CSS / MPEG-7) ───────────────────────────────────

def curvature_scale_space(
    curve: list[tuple[float, float]],
    sigmas: list[float] | None = None
) -> list[tuple[float, list[float]]]:
    """
    CSS по MPEG-7 (Mokhtarian et al.).
    Возвращает [(sigma, zero_crossings)] для нескольких масштабов.
    """
    if sigmas is None:
        sigmas = [1.0, 2.0, 4.0, 8.0]

    result = []
    for sigma in sigmas:
        zc = _zero_crossings_at_sigma(curve, sigma)
        result.append((sigma, zc))
    return result


def _zero_crossings_at_sigma(
    curve: list[tuple[float, float]],
    sigma: float
) -> list[float]:
    n = len(curve)
    if n < 3:
        return []

    xs = _gaussian_smooth([p[0] for p in curve], sigma)
    ys = _gaussian_smooth([p[1] for p in curve], sigma)

    kappa = []
    for i in range(n):
        im1 = (i - 1) % n
        ip1 = (i + 1) % n
        dx  = xs[ip1] - xs[im1]
        dy  = ys[ip1] - ys[im1]
        ddx = xs[ip1] - 2*xs[i] + xs[im1]
        ddy = ys[ip1] - 2*ys[i] + ys[im1]
        denom = (dx**2 + dy**2)**1.5 + 1e-10
        kappa.append((dx*ddy - dy*ddx) / denom)

    zc = []
    for i in range(n):
        j = (i + 1) % n
        if kappa[i] * kappa[j] < 0:
            t = kappa[i] / (kappa[i] - kappa[j] + 1e-10)
            zc.append((i + t) / n)
    return zc


def _gaussian_smooth(arr: list[float], sigma: float) -> list[float]:
    radius = max(1, int(3 * sigma))
    kernel = [math.exp(-0.5*(k/sigma)**2) for k in range(-radius, radius+1)]
    s = sum(kernel)
    kernel = [k/s for k in kernel]
    n = len(arr)
    result = []
    for i in range(n):
        val = sum(kernel[j] * arr[(i + j - radius) % n]
                  for j in range(len(kernel)))
        result.append(val)
    return result


def freeman_chain_code(curve: list[tuple[float, float]]) -> str:
    """8-направленный цепной код Фримана — быстрый хэш формы кривой."""
    dirs = {
        ( 1,  0): '0', ( 1,  1): '1', ( 0,  1): '2', (-1,  1): '3',
        (-1,  0): '4', (-1, -1): '5', ( 0, -1): '6', ( 1, -1): '7',
    }
    code = []
    for i in range(len(curve) - 1):
        dx = curve[i+1][0] - curve[i][0]
        dy = curve[i+1][1] - curve[i][1]
        sx = (1 if dx>0 else -1 if dx<0 else 0)
        sy = (1 if dy>0 else -1 if dy<0 else 0)
        code.append(dirs.get((sx, sy), '?'))
    return ''.join(code)


# ── главная функция ─────────────────────────────────────────────────────────

def build_fractal_signature(
    boundary_curve: list[tuple[float, float]]
) -> FractalSignature:
    """Построить FractalSignature из граничной кривой между сообществами."""
    return FractalSignature(
        fd_box     = box_counting_dimension(boundary_curve),
        fd_divider = divider_dimension(boundary_curve),
        ifs_coeffs = fit_ifs_coefficients(boundary_curve),
        css_image  = curvature_scale_space(boundary_curve),
        chain_code = freeman_chain_code(boundary_curve),
        curve      = boundary_curve,
    )
