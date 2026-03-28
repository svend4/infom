"""
ASCII визуализация графа знаний в терминале.
Stdlib only, работает везде.
"""
from __future__ import annotations
import math

from graph import KnowledgeMap, Community
from signatures import ShapeClass

# ── символы форм ─────────────────────────────────────────────────────────────

SHAPE_SYMBOL = {
    ShapeClass.TRIANGLE:   "△",
    ShapeClass.RECTANGLE:  "□",
    ShapeClass.TRAPEZOID:  "⊡",
    ShapeClass.PENTAGON:   "⬠",
    ShapeClass.HEXAGON:    "⬡",
    ShapeClass.HEPTAGON:   "☆",
    ShapeClass.OCTAGON:    "✳",
    ShapeClass.POLYGON:    "◇",
}

SHAPE_COLOR_CODE = {
    ShapeClass.TRIANGLE:   "\033[93m",   # жёлтый
    ShapeClass.RECTANGLE:  "\033[96m",   # голубой
    ShapeClass.TRAPEZOID:  "\033[36m",   # циан
    ShapeClass.PENTAGON:   "\033[92m",   # зелёный
    ShapeClass.HEXAGON:    "\033[94m",   # синий
    ShapeClass.HEPTAGON:   "\033[95m",   # маджента
    ShapeClass.OCTAGON:    "\033[91m",   # красный
    ShapeClass.POLYGON:    "\033[37m",   # белый
}

COMMUNITY_COLORS = [
    "\033[44m", "\033[42m", "\033[43m", "\033[45m",
    "\033[46m", "\033[41m", "\033[100m", "\033[103m",
]
RESET = "\033[0m"
BOLD  = "\033[1m"
DIM   = "\033[2m"


def _color(text: str, code: str) -> str:
    return f"{code}{text}{RESET}"


def render_legend() -> str:
    lines = [
        f"{BOLD}Геометрическая иерархия:{RESET}",
        f"  {SHAPE_COLOR_CODE[ShapeClass.TRIANGLE]}△{RESET} Треугольник  (3)  — базовая триада    [meta2/Tangram]",
        f"  {SHAPE_COLOR_CODE[ShapeClass.RECTANGLE]}□{RESET} Прямоугольник(4)  — устойч. паттерн  [meta2/Tangram]",
        f"  {SHAPE_COLOR_CODE[ShapeClass.PENTAGON]}⬠{RESET} Пятиугольник (5)  — сложн. кластер   [meta2/Tangram]",
        f"  {SHAPE_COLOR_CODE[ShapeClass.HEXAGON]}⬡{RESET} Шестиугольник(6)  — Q6-сота           [meta/hexcore]",
        f"  {SHAPE_COLOR_CODE[ShapeClass.HEPTAGON]}☆{RESET} 7-лучевая    (7)  — 3D звезда магов   [infom]",
        f"  {SHAPE_COLOR_CODE[ShapeClass.OCTAGON]}✳{RESET} 8-лучевая    (8)  — 3D роза ветров    [infom]",
        f"  {DIM}∿  Фрактал      (N)  — граница групп    [meta2/Fractal]{RESET}",
    ]
    return "\n".join(lines)


def render_communities(km: KnowledgeMap) -> str:
    """Таблица сообществ с геометрическими подписями."""
    if not km.communities:
        return "Сообществ нет."

    lp_info = ""
    if km.lp_iterations:
        mod_q   = f"  modularity Q={km.modularity:.3f}" if km.modularity else ""
        lp_info = f"  {DIM}(LP: {km.lp_iterations} iter{mod_q}){RESET}"
    lines = [f"{BOLD}Сообщества ({len(km.communities)}):{RESET}{lp_info}"]
    for i, comm in enumerate(km.communities.values()):
        cc   = COMMUNITY_COLORS[i % len(COMMUNITY_COLORS)]
        tag  = _color(f" {comm.id[-4:]} ", cc)

        shape = comm.tangram.shape_class if comm.tangram else None
        ssym  = SHAPE_SYMBOL.get(shape, "?")
        scol  = SHAPE_COLOR_CODE.get(shape, "")
        sstr  = _color(ssym, scol) if scol else ssym

        skel = comm.octagram.skeleton_type.value[:5] if comm.octagram else "?"
        hid  = comm.hex_id
        bits = "".join(str(b) for b in comm.hex_sig.bits) if comm.hex_sig else "??????"

        node_labels = [n.label for n in comm.nodes]
        nodes_str   = ", ".join(node_labels[:5])
        if len(node_labels) > 5:
            nodes_str += f" +{len(node_labels)-5}"

        # фрактальные границы
        borders  = [b for b in km.borders
                    if b.community_a == comm.id or b.community_b == comm.id]
        fd_str   = ""
        if borders:
            fd_avg = sum(b.fractal.fd_box for b in borders) / len(borders)
            bar    = "▓" if fd_avg > 1.5 else "░"
            fd_str = f" fd={fd_avg:.2f}{bar}"

        arch_tag = f" \033[2m{comm.dominant_archetype}\033[0m" if comm.nodes else ""
        lines.append(
            f"  {tag}{arch_tag} {sstr} {scol}shape={shape.value if shape else '?':12s}{RESET}"
            f" Q6={hid:2d}[{bits}] skel={skel:5s}{fd_str}"
            f"  [{nodes_str}]"
        )
    return "\n".join(lines)


def render_hyper_edges(km: KnowledgeMap) -> str:
    """Список гиперрёбер с Tangram + Heptagram."""
    if not km.hyper_edges:
        return "Гиперрёбер нет."

    lines = [f"{BOLD}Гиперрёбра ({len(km.hyper_edges)}):{RESET}"]
    for he in km.hyper_edges:
        shape = he.tangram.shape_class if he.tangram else None
        ssym  = SHAPE_SYMBOL.get(shape, "◇")
        scol  = SHAPE_COLOR_CODE.get(shape, "")
        sstr  = _color(f"{ssym} {shape.value if shape else '?':12s}", scol)

        dom = ""
        if he.heptagram:
            r   = he.heptagram.dominant_ray
            bar = "█" * int(r.length * 8) + "░" * (8 - int(r.length * 8))
            dom = f" [{r.label:10s} {bar}]"

        node_labels = [km.nodes[n].label for n in he.nodes if n in km.nodes]
        nodes_str   = " · ".join(node_labels)

        lines.append(f"  {sstr} n={he.n_nodes}{dom}  {nodes_str}")
    return "\n".join(lines)


def render_fractal_borders(km: KnowledgeMap) -> str:
    """Фрактальные границы между сообществами."""
    if not km.borders:
        return "Границ нет."

    lines = [f"{BOLD}Фрактальные границы ({len(km.borders)}):{RESET}"]
    for b in km.borders:
        fd  = b.fractal.fd_box
        fdd = b.fractal.fd_divider
        ifs = b.fractal.ifs_coeffs[:3]
        tag = _color("размытая", "\033[91m") if fd > 1.5 else _color("чёткая  ", "\033[92m")

        bar_len = min(20, int(fd / 2.0 * 20))
        bar     = "▓" * bar_len + "░" * (20 - bar_len)

        ifs_str = " ".join(f"{c:+.2f}" for c in ifs)
        lines.append(
            f"  {b.community_a[-4:]} ↔ {b.community_b[-4:]}"
            f"  fd_box={fd:.3f} [{bar}] {tag}"
            f"  ifs=[{ifs_str}]"
        )
    return "\n".join(lines)


def render_graph_ascii(km: KnowledgeMap, width: int = 60) -> str:
    """
    ASCII-сетка: ноды расставлены по PCA-координатам (глобальная проекция).
    Рёбра показаны символом · в средней точке.
    """
    if not km.nodes:
        return ""

    nodes = list(km.nodes.values())
    n     = len(nodes)
    H     = max(12, n + 2)
    W     = width

    # используем PCA-позиции из KnowledgeMap если доступны
    raw_pos = km.pca_positions if km.pca_positions else {
        nd.id: (nd.embedding[0] if nd.embedding else 0.0,
                nd.embedding[1] if len(nd.embedding) > 1 else 0.0)
        for nd in nodes
    }

    # нормализуем в [0, W-7] × [0, H-2]
    xs = [p[0] for p in raw_pos.values()]
    ys = [p[1] for p in raw_pos.values()]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    dx = (x_max - x_min) or 1.0
    dy = (y_max - y_min) or 1.0

    def to_grid(nid: str) -> tuple[int, int]:
        px, py = raw_pos.get(nid, (0.0, 0.0))
        gx = int((px - x_min) / dx * (W - 8))
        gy = int((py - y_min) / dy * (H - 2))
        return gx, gy

    grid = [[" "] * W for _ in range(H)]
    node_pos: dict[str, tuple[int, int]] = {}

    for node in nodes:
        x, y = to_grid(node.id)
        y = max(0, min(H - 1, y))
        # разрешаем коллизии сдвигом по x
        while grid[y][x] != " " and x < W - 7:
            x += 1
        label = node.label[:6]
        for j, ch in enumerate(label):
            if x + j < W:
                grid[y][x + j] = ch
        node_pos[node.id] = (x + len(label) // 2, y)

    # рёбра — точка в середине между узлами
    for edge in km.edges[:30]:
        if edge.source not in node_pos or edge.target not in node_pos:
            continue
        x0, y0 = node_pos[edge.source]
        x1, y1 = node_pos[edge.target]
        mx, my = (x0 + x1) // 2, (y0 + y1) // 2
        if 0 <= my < H and 0 <= mx < W and grid[my][mx] == " ":
            # символ ребра зависит от веса
            grid[my][mx] = "·" if edge.weight < 0.7 else "•"

    # модулярность в заголовке
    mod_str = f"  Q={km.modularity:.3f}" if km.modularity != 0.0 else ""
    lines = [f"{BOLD}Граф (PCA-проекция, {n} нод, {len(km.edges)} рёбер{mod_str}):{RESET}"]
    lines.append("┌" + "─" * W + "┐")
    for row in grid:
        lines.append("│" + "".join(row) + "│")
    lines.append("└" + "─" * W + "┘")
    return "\n".join(lines)


def render_heptagram_ascii(km: KnowledgeMap, community_id: str) -> str:
    """Спайдер-диаграмма HeptagramSignature для сообщества."""
    comm = km.communities.get(community_id)
    if not comm or not comm.heptagram:
        return ""

    hept  = comm.heptagram
    lines = [f"{BOLD}Heptagram [{comm.label}]{RESET}  "
             f"симметрия={hept.symmetry_score:.2f}  "
             f"энергия={hept.total_energy:.2f}"]

    for ray in hept.rays:
        bar_len = int(ray.length * 20)
        bar     = "█" * bar_len + "░" * (20 - bar_len)
        dom_mark = " ◄ dominant" if ray is hept.dominant_ray else ""
        lines.append(
            f"  {ray.label:10s} [{bar}] {ray.length:.2f}"
            f"  z={ray.z:+.2f}{dom_mark}"
        )
    return "\n".join(lines)


def render_octagram_ascii(km: KnowledgeMap, community_id: str) -> str:
    """Роза ветров OctagramSignature для сообщества."""
    comm = km.communities.get(community_id)
    if not comm or not comm.octagram:
        return ""

    oct_  = comm.octagram
    rays  = {r.direction: r for r in oct_.rays}
    skel  = oct_.skeleton_type.value

    # ASCII роза 5×5
    def bar(d: str) -> str:
        r = rays.get(d)
        if not r:
            return "·"
        n = int(r.length * 3)
        return ["·", "░", "▒", "▓"][min(n, 3)]

    N   = bar("N");  NE = bar("NE"); E  = bar("E");  SE = bar("SE")
    S   = bar("S");  SW = bar("SW"); W  = bar("W");  NW = bar("NW")

    lines = [
        f"{BOLD}Octagram [{comm.label}]{RESET}  скелет={skel}",
        f"       {NW} {N} {NE}",
        f"        ↖↑↗",
        f"    {W} ←─●─→ {E}",
        f"        ↙↓↘",
        f"       {SW} {S} {SE}",
    ]
    ax1, ax2 = oct_.dominant_axis
    lines.append(f"  доминирующая ось: {ax1} ↔ {ax2}")
    return "\n".join(lines)


def render_q6_map(km: KnowledgeMap) -> str:
    """Карта Q6 — показывает все 64 позиции, занятые отмечает."""
    occupied: dict[int, str] = {}
    for comm in km.communities.values():
        occupied[comm.hex_id] = comm.label[:3]
    for node in km.nodes.values():
        if node.hex_id not in occupied:
            occupied[node.hex_id] = "·" + node.label[:2]

    lines = [f"{BOLD}Q6 карта (64 позиции, занято {len(occupied)}):{RESET}"]
    lines.append("    " + " ".join(f"{i:2d}" for i in range(8)))
    lines.append("   ┌" + "───" * 8 + "┐")
    for row in range(8):
        cells = []
        for col in range(8):
            hid = row * 8 + col
            if hid in occupied:
                cells.append(f"\033[93m{occupied[hid][:2]:>2s}\033[0m")
            else:
                cells.append(" ·")
        lines.append(f"{row:2d} │" + " ".join(cells) + " │")
    lines.append("   └" + "───" * 8 + "┘")
    return "\n".join(lines)


# ── главная функция ─────────────────────────────────────────────────────────

def render_full(km: KnowledgeMap, width: int = 70) -> str:
    sep = "─" * width
    parts = [
        render_legend(),
        sep,
        render_graph_ascii(km, width),
        sep,
        render_communities(km),
        sep,
        render_hyper_edges(km),
        sep,
        render_fractal_borders(km),
        sep,
        render_q6_map(km),
    ]
    # heptagram + octagram для первого сообщества
    if km.communities:
        first_id = next(iter(km.communities))
        h = render_heptagram_ascii(km, first_id)
        o = render_octagram_ascii(km, first_id)
        if h:
            parts += [sep, h]
        if o:
            parts += [sep, o]

    return "\n".join(parts)
