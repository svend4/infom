"""
HTML/D3.js интерактивная визуализация графа знаний.
Генерирует self-contained HTML файл — открывается в любом браузере.

Показывает:
  - Force-directed граф (ноды + рёбра)
  - Цветные выпуклые оболочки сообществ
  - Tangram полигоны для гиперрёбер (△□⬠⬡☆✳)
  - Фрактальные кривые границ между сообществами
  - Heptagram спайдер-диаграмма
  - Octagram роза ветров (3D скелет)
  - Q6 карта как шестиугольная сота
"""
from __future__ import annotations
import json
import math
from graph import KnowledgeMap


# ── сериализация данных ───────────────────────────────────────────────────────

def _km_to_json(km: KnowledgeMap) -> str:
    """Сериализует KnowledgeMap в JSON для D3."""

    # ноды
    nodes = []
    for node in km.nodes.values():
        comm = km.find_community(node.id)
        nodes.append({
            "id":        node.id,
            "label":     node.label,
            "archetype": node.archetype,
            "hex_id":    node.hex_id,
            "bits":      list(node.hex_sig.bits) if node.hex_sig else [],
            "community": comm.id if comm else None,
            "weight":    node.weight,
        })

    # рёбра
    edges = []
    for e in km.edges:
        edges.append({
            "source":   e.source,
            "target":   e.target,
            "label":    e.label,
            "weight":   e.weight,
            "directed": e.directed,
        })

    # сообщества
    communities = []
    for i, comm in enumerate(km.communities.values()):
        shape  = comm.tangram.shape_class.value if comm.tangram else "unknown"
        skel   = comm.octagram.skeleton_type.value if comm.octagram else "unknown"
        hept   = []
        if comm.heptagram:
            hept = [{"label": r.label, "length": r.length,
                     "angle": r.angle, "z": r.z}
                    for r in comm.heptagram.rays]
        octa = []
        if comm.octagram:
            octa = [{"dir": r.direction, "length": r.length,
                     "elevation": r.elevation}
                    for r in comm.octagram.rays]
        communities.append({
            "id":       comm.id,
            "label":    comm.label,
            "hex_id":   comm.hex_id,
            "bits":     list(comm.hex_sig.bits) if comm.hex_sig else [],
            "shape":    shape,
            "skeleton": skel,
            "color_idx": i,
            "node_ids": [n.id for n in comm.nodes],
            "heptagram": hept,
            "octagram":  octa,
            "fd_box":    comm.fractal.fd_box if comm.fractal else 1.0,
        })

    # гиперрёбра
    hyper_edges = []
    for he in km.hyper_edges:
        shape = he.tangram.shape_class.value if he.tangram else "unknown"
        hyper_edges.append({
            "id":     he.id,
            "label":  he.label,
            "shape":  shape,
            "nodes":  he.nodes,
            "weight": he.weight,
        })

    # границы
    borders = []
    for b in km.borders:
        borders.append({
            "a":         b.community_a,
            "b":         b.community_b,
            "fd_box":    b.fractal.fd_box,
            "fd_divider": b.fractal.fd_divider,
            "ifs":       b.fractal.ifs_coeffs[:4],
            "curve":     [(x, y) for x, y in b.fractal.curve[:12]],
        })

    return json.dumps({
        "nodes":       nodes,
        "edges":       edges,
        "communities": communities,
        "hyper_edges": hyper_edges,
        "borders":     borders,
    }, ensure_ascii=False, indent=2)


# ── HTML шаблон ───────────────────────────────────────────────────────────────

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>InfoM — Knowledge Map</title>
<script src="https://cdn.jsdelivr.net/npm/d3@7/dist/d3.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d1117; color: #c9d1d9;
  font-family: 'Segoe UI', monospace; font-size: 13px;
}
#app { display: flex; height: 100vh; }
#sidebar {
  width: 300px; min-width: 260px;
  background: #161b22; border-right: 1px solid #30363d;
  overflow-y: auto; padding: 12px;
}
#main { flex: 1; position: relative; }
svg { width: 100%; height: 100%; }

/* sidebar */
h2 { color: #58a6ff; font-size: 14px; margin: 10px 0 6px; border-bottom: 1px solid #30363d; padding-bottom: 4px; }
h3 { color: #8b949e; font-size: 12px; margin: 8px 0 4px; }
.comm-card {
  background: #21262d; border-radius: 6px; padding: 8px;
  margin-bottom: 6px; border-left: 3px solid;
  cursor: pointer; transition: background 0.2s;
}
.comm-card:hover { background: #2d333b; }
.comm-card.active { background: #2d333b; }
.tag {
  display: inline-block; padding: 1px 6px; border-radius: 10px;
  font-size: 11px; margin: 2px 2px 2px 0;
  background: #21262d; border: 1px solid #30363d;
}
.shape-tag { color: #ffa657; border-color: #ffa657; }
.skel-tag  { color: #79c0ff; border-color: #79c0ff; }
.q6-tag    { color: #3fb950; border-color: #3fb950; }

/* spider / rose charts */
.chart-wrap { background: #21262d; border-radius: 8px; padding: 8px; margin: 6px 0; }
.chart-title { color: #8b949e; font-size: 11px; margin-bottom: 4px; }

/* graph elements */
.node circle {
  stroke-width: 2;
  transition: r 0.2s;
}
.node text {
  pointer-events: none;
  font-size: 11px;
  fill: #c9d1d9;
  text-shadow: 0 0 3px #0d1117;
}
.link {
  stroke: #444c56;
  stroke-opacity: 0.6;
}
.link.directed marker { fill: #444c56; }
.hull {
  opacity: 0.12;
  stroke-width: 2;
  stroke-opacity: 0.5;
}
.hyper-poly {
  fill: none;
  stroke-width: 1.5;
  stroke-dasharray: 4 2;
  opacity: 0.7;
}
.fractal-border {
  fill: none;
  stroke-width: 1;
  stroke-opacity: 0.4;
  stroke-dasharray: 2 4;
}
.tooltip {
  position: absolute; background: #161b22;
  border: 1px solid #30363d; border-radius: 6px;
  padding: 8px 12px; pointer-events: none;
  font-size: 12px; max-width: 200px;
  opacity: 0; transition: opacity 0.15s;
}

/* legend */
#legend {
  position: absolute; bottom: 16px; left: 16px;
  background: #161b22cc; border: 1px solid #30363d;
  border-radius: 8px; padding: 10px; font-size: 12px;
}
#legend div { margin: 3px 0; }
</style>
</head>
<body>
<div id="app">
<div id="sidebar">
  <h2>InfoM Knowledge Map</h2>
  <div id="stats"></div>
  <h2>Сообщества</h2>
  <div id="comm-list"></div>
  <h2>Подписи</h2>
  <div id="sig-panel">
    <div class="chart-wrap">
      <div class="chart-title">Heptagram (7 измерений)</div>
      <svg id="spider" width="276" height="200"></svg>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Octagram (роза ветров)</div>
      <svg id="rose" width="276" height="200"></svg>
    </div>
    <div class="chart-wrap">
      <div class="chart-title">Q6 соты</div>
      <svg id="q6map" width="276" height="130"></svg>
    </div>
  </div>
</div>
<div id="main">
  <div class="tooltip" id="tooltip"></div>
  <div id="legend">
    <div>△ Треугольник  (3) triada</div>
    <div>□ Прямоуг.    (4) pattern</div>
    <div>⬠ Пятиугольн. (5) cluster</div>
    <div>⬡ Шестиугольн.(6) Q6-cell</div>
    <div>☆ 7-лучевая   (7) 3D heptagram</div>
    <div>✳ 8-лучевая   (8) 3D octagram</div>
    <div>∿ Фрактал     (N) border</div>
  </div>
  <svg id="graph"></svg>
</div>
</div>

<script>
const DATA = __DATA__;

// ── цвета ──────────────────────────────────────────────────────────────────
const PALETTE = [
  "#58a6ff","#3fb950","#ffa657","#f78166",
  "#d2a8ff","#79c0ff","#56d364","#ff7b72",
];

const SHAPE_SYMBOL = {
  triangle:"△", rectangle:"□", trapezoid:"⊡",
  pentagon:"⬠", hexagon:"⬡", heptagon:"☆",
  octagon:"✳", polygon:"◇", unknown:"·"
};

const commColor = id => {
  const idx = DATA.communities.findIndex(c => c.id === id);
  return PALETTE[idx % PALETTE.length];
};

// ── sidebar ────────────────────────────────────────────────────────────────
document.getElementById("stats").innerHTML = `
  <div class="tag">${DATA.nodes.length} нод</div>
  <div class="tag">${DATA.edges.length} рёбер</div>
  <div class="tag">${DATA.hyper_edges.length} гиперрёбер</div>
  <div class="tag">${DATA.communities.length} сообществ</div>
  <div class="tag">${DATA.borders.length} границ</div>
`;

let activeComm = null;

function renderCommList() {
  const el = document.getElementById("comm-list");
  el.innerHTML = DATA.communities.map(c => {
    const sym = SHAPE_SYMBOL[c.shape] || "·";
    const col = PALETTE[c.color_idx % PALETTE.length];
    return `<div class="comm-card ${activeComm===c.id?'active':''}"
      style="border-color:${col}"
      onclick="selectComm('${c.id}')">
      <span style="color:${col};font-size:16px">${sym}</span>
      <strong style="color:${col}"> ${c.label}</strong><br>
      <span class="tag shape-tag">${c.shape}</span>
      <span class="tag skel-tag">${c.skeleton}</span>
      <span class="tag q6-tag">Q6=${c.hex_id}</span>
      <div style="color:#8b949e;font-size:11px;margin-top:4px">
        ${c.node_ids.length} нод  fd=${c.fd_box.toFixed(3)}
      </div>
    </div>`;
  }).join("");
}
renderCommList();

function selectComm(id) {
  activeComm = id;
  renderCommList();
  const c = DATA.communities.find(x => x.id === id);
  if (c) {
    drawSpider(c.heptagram);
    drawRose(c.octagram);
    drawQ6Map(c.hex_id);
  }
}
if (DATA.communities.length) selectComm(DATA.communities[0].id);

// ── spider chart ──────────────────────────────────────────────────────────
function drawSpider(rays) {
  const svg = d3.select("#spider");
  svg.selectAll("*").remove();
  if (!rays || !rays.length) return;
  const w = 276, h = 200, cx = w/2, cy = h/2, r = 80;
  const n = rays.length;
  const angle = i => (i / n) * 2 * Math.PI - Math.PI/2;

  // сетка
  [0.25,0.5,0.75,1.0].forEach(t => {
    const pts = rays.map((_, i) => {
      const a = angle(i);
      return [cx + Math.cos(a)*r*t, cy + Math.sin(a)*r*t];
    });
    svg.append("polygon")
      .attr("points", pts.map(p=>p.join(",")).join(" "))
      .attr("fill","none").attr("stroke","#30363d").attr("stroke-width",1);
  });

  // оси
  rays.forEach((_, i) => {
    const a = angle(i);
    svg.append("line")
      .attr("x1",cx).attr("y1",cy)
      .attr("x2",cx+Math.cos(a)*r).attr("y2",cy+Math.sin(a)*r)
      .attr("stroke","#30363d").attr("stroke-width",1);
  });

  // данные
  const pts = rays.map((ray, i) => {
    const a = angle(i);
    return [cx + Math.cos(a)*r*ray.length, cy + Math.sin(a)*r*ray.length];
  });
  svg.append("polygon")
    .attr("points", pts.map(p=>p.join(",")).join(" "))
    .attr("fill","#58a6ff").attr("fill-opacity",0.25)
    .attr("stroke","#58a6ff").attr("stroke-width",2);

  pts.forEach((p,i) => {
    svg.append("circle").attr("cx",p[0]).attr("cy",p[1]).attr("r",3)
      .attr("fill","#58a6ff");
  });

  // метки
  rays.forEach((ray, i) => {
    const a = angle(i);
    const lx = cx + Math.cos(a)*(r+14);
    const ly = cy + Math.sin(a)*(r+14);
    svg.append("text")
      .attr("x",lx).attr("y",ly)
      .attr("text-anchor","middle").attr("dominant-baseline","middle")
      .attr("fill","#8b949e").attr("font-size",10)
      .text(ray.label.slice(0,6));
  });
}

// ── rose chart ────────────────────────────────────────────────────────────
function drawRose(rays) {
  const svg = d3.select("#rose");
  svg.selectAll("*").remove();
  if (!rays || !rays.length) return;
  const w = 276, h = 200, cx = w/2, cy = h/2, r = 80;
  const dirs = ["N","NE","E","SE","S","SW","W","NW"];
  const n = 8;
  const angle = i => (i / n) * 2 * Math.PI - Math.PI/2;

  [0.25,0.5,0.75,1.0].forEach(t => {
    const pts = dirs.map((_,i)=>{
      const a=angle(i);
      return [cx+Math.cos(a)*r*t, cy+Math.sin(a)*r*t];
    });
    svg.append("polygon")
      .attr("points",pts.map(p=>p.join(",")).join(" "))
      .attr("fill","none").attr("stroke","#30363d").attr("stroke-width",1);
  });
  dirs.forEach((_,i)=>{
    const a=angle(i);
    svg.append("line")
      .attr("x1",cx).attr("y1",cy)
      .attr("x2",cx+Math.cos(a)*r).attr("y2",cy+Math.sin(a)*r)
      .attr("stroke","#30363d").attr("stroke-width",1);
  });

  const rayMap = {};
  rays.forEach(r2 => { rayMap[r2.dir] = r2; });

  const pts = dirs.map((d,i)=>{
    const ray = rayMap[d] || {length:0};
    const a   = angle(i);
    // 3D elevation меняет радиус
    const el  = ray.elevation || 0;
    const rr  = r * ray.length * Math.cos(el);
    return [cx+Math.cos(a)*rr, cy+Math.sin(a)*rr];
  });
  svg.append("polygon")
    .attr("points",pts.map(p=>p.join(",")).join(" "))
    .attr("fill","#ffa657").attr("fill-opacity",0.2)
    .attr("stroke","#ffa657").attr("stroke-width",2);

  dirs.forEach((d,i)=>{
    const a = angle(i);
    const lx = cx+Math.cos(a)*(r+14);
    const ly = cy+Math.sin(a)*(r+14);
    svg.append("text")
      .attr("x",lx).attr("y",ly)
      .attr("text-anchor","middle").attr("dominant-baseline","middle")
      .attr("fill","#8b949e").attr("font-size",10).text(d);
  });
}

// ── Q6 honeycomb ──────────────────────────────────────────────────────────
function drawQ6Map(activeHexId) {
  const svg = d3.select("#q6map");
  svg.selectAll("*").remove();
  const W = 276, H = 130;
  // 8×8 сетка
  const cellW = W/8, cellH = H/8;
  const occupied = new Set(DATA.communities.map(c=>c.hex_id));
  DATA.nodes.forEach(n => occupied.add(n.hex_id));

  for (let row=0; row<8; row++) {
    for (let col=0; col<8; col++) {
      const hid = row*8+col;
      const x   = col*cellW + cellW/2;
      const y   = row*cellH + cellH/2;
      const comm = DATA.communities.find(c=>c.hex_id===hid);
      const isActive = hid === activeHexId;
      const fill = comm ? PALETTE[comm.color_idx%PALETTE.length]
                        : (occupied.has(hid) ? "#444c56" : "#21262d");
      const opacity = comm ? 0.8 : (occupied.has(hid) ? 0.5 : 0.3);
      svg.append("rect")
        .attr("x",col*cellW+1).attr("y",row*cellH+1)
        .attr("width",cellW-2).attr("height",cellH-2)
        .attr("rx",2)
        .attr("fill",fill).attr("fill-opacity",opacity)
        .attr("stroke", isActive?"#fff":"none")
        .attr("stroke-width", isActive?2:0);
      if (comm || isActive) {
        svg.append("text")
          .attr("x",x).attr("y",y+1)
          .attr("text-anchor","middle").attr("dominant-baseline","middle")
          .attr("fill","#fff").attr("font-size",8)
          .text(hid);
      }
    }
  }
}

// ── force graph ────────────────────────────────────────────────────────────
const svg   = d3.select("#graph");
const w     = () => svg.node().clientWidth;
const h     = () => svg.node().clientHeight;

svg.append("defs").append("marker")
  .attr("id","arrow").attr("viewBox","0 -5 10 10")
  .attr("refX",18).attr("refY",0)
  .attr("markerWidth",6).attr("markerHeight",6)
  .attr("orient","auto")
  .append("path").attr("d","M0,-5L10,0L0,5").attr("fill","#555");

const g = svg.append("g");

// zoom + pan
svg.call(d3.zoom().scaleExtent([0.2, 4])
  .on("zoom", e => g.attr("transform", e.transform)));

// данные для симуляции
const simNodes = DATA.nodes.map(n => ({...n}));
const nodeMap  = Object.fromEntries(simNodes.map(n => [n.id, n]));

const simLinks = DATA.edges
  .filter(e => nodeMap[e.source] && nodeMap[e.target])
  .map(e => ({...e, source: nodeMap[e.source], target: nodeMap[e.target]}));

// community hulls (выпуклые оболочки)
const hullG = g.append("g").attr("class","hulls");
// hyper-edge polygons
const polyG = g.append("g").attr("class","polys");
// fractal borders
const borderG = g.append("g").attr("class","borders");
// links
const linkG = g.append("g");
// nodes
const nodeG = g.append("g");

const link = linkG.selectAll("line").data(simLinks).join("line")
  .attr("class","link")
  .attr("stroke-width", d => Math.sqrt(d.weight)*2)
  .attr("marker-end", d => d.directed ? "url(#arrow)" : null);

const node = nodeG.selectAll("g").data(simNodes).join("g")
  .attr("class","node")
  .call(d3.drag()
    .on("start", (e,d) => { if (!e.active) sim.alphaTarget(0.3).restart(); d.fx=d.x; d.fy=d.y; })
    .on("drag",  (e,d) => { d.fx=e.x; d.fy=e.y; })
    .on("end",   (e,d) => { if (!e.active) sim.alphaTarget(0); d.fx=null; d.fy=null; }));

node.append("circle")
  .attr("r", 16)
  .attr("fill", d => { const c=DATA.communities.find(c=>c.id===d.community); return c?PALETTE[c.color_idx%PALETTE.length]:"#444c56"; })
  .attr("stroke", d => { const c=DATA.communities.find(c=>c.id===d.community); return c?PALETTE[c.color_idx%PALETTE.length]:"#666"; })
  .on("mouseover", (e,d) => showTooltip(e, `<b>${d.label}</b><br>arch: ${d.archetype}<br>Q6: ${d.hex_id} [${d.bits.join("")}]`))
  .on("mouseout",  () => hideTooltip())
  .on("click",     (e,d) => { const comm = DATA.communities.find(c=>c.id===d.community); if(comm) selectComm(comm.id); });

node.append("text")
  .attr("dy","0.35em").attr("text-anchor","middle")
  .text(d => d.label.slice(0,8));

// simulation
const sim = d3.forceSimulation(simNodes)
  .force("link",    d3.forceLink(simLinks).id(d=>d.id).distance(80).strength(0.5))
  .force("charge",  d3.forceManyBody().strength(-200))
  .force("center",  d3.forceCenter(600, 400))
  .force("collide", d3.forceCollide(30))
  .on("tick", ticked);

function ticked() {
  link
    .attr("x1", d=>d.source.x).attr("y1", d=>d.source.y)
    .attr("x2", d=>d.target.x).attr("y2", d=>d.target.y);

  node.attr("transform", d=>`translate(${d.x},${d.y})`);

  drawHulls();
  drawHyperPolys();
  drawBorders();
}

function drawHulls() {
  hullG.selectAll("path").remove();
  DATA.communities.forEach(c => {
    const pts = c.node_ids
      .map(id => nodeMap[id])
      .filter(Boolean)
      .map(n => [n.x, n.y]);
    if (pts.length < 2) return;
    const col = PALETTE[c.color_idx % PALETTE.length];
    try {
      const hull = pts.length > 2 ? d3.polygonHull(pts) : pts;
      if (!hull) return;
      const padded = hull.map(([x,y]) => {
        const cx = d3.mean(hull,p=>p[0]), cy = d3.mean(hull,p=>p[1]);
        const dx=x-cx, dy=y-cy, len=Math.hypot(dx,dy)||1;
        return [x + dx/len*20, y + dy/len*20];
      });
      hullG.append("path")
        .attr("d", "M" + padded.map(p=>p.join(",")).join("L") + "Z")
        .attr("class","hull")
        .attr("fill",col).attr("stroke",col);
    } catch(ex) {}
  });
}

const SHAPE_COLORS = {
  triangle:"#ffd700", rectangle:"#00bfff",
  pentagon:"#7fff00", hexagon:"#da70d6",
  heptagon:"#ff69b4", octagon:"#ff4500",
  polygon:"#20b2aa", unknown:"#666"
};

function drawHyperPolys() {
  polyG.selectAll("path").remove();
  DATA.hyper_edges.forEach(he => {
    const pts = he.nodes
      .map(id => nodeMap[id]).filter(Boolean)
      .map(n => [n.x, n.y]);
    if (pts.length < 3) return;
    try {
      const hull = d3.polygonHull(pts); if (!hull) return;
      polyG.append("path")
        .attr("d","M"+hull.map(p=>p.join(",")).join("L")+"Z")
        .attr("class","hyper-poly")
        .attr("stroke", SHAPE_COLORS[he.shape]||"#666");
    } catch(ex) {}
  });
}

function drawBorders() {
  borderG.selectAll("path").remove();
  DATA.borders.forEach(b => {
    const ca = DATA.communities.find(c=>c.id===b.a);
    const cb = DATA.communities.find(c=>c.id===b.b);
    if (!ca || !cb) return;
    const ptsA = ca.node_ids.map(id=>nodeMap[id]).filter(Boolean);
    const ptsB = cb.node_ids.map(id=>nodeMap[id]).filter(Boolean);
    if (!ptsA.length || !ptsB.length) return;
    const cx1 = d3.mean(ptsA,n=>n.x), cy1 = d3.mean(ptsA,n=>n.y);
    const cx2 = d3.mean(ptsB,n=>n.x), cy2 = d3.mean(ptsB,n=>n.y);
    const fd  = b.fd_box;
    // фрактальная кривая — синусоидальное возмущение прямой
    const pts = [];
    const N = 12;
    for (let i=0;i<=N;i++) {
      const t  = i/N;
      const x  = cx1 + t*(cx2-cx1);
      const y  = cy1 + t*(cy2-cy1);
      const dx = -(cy2-cy1), dy = cx2-cx1;
      const len = Math.hypot(dx,dy)||1;
      const amp = fd*15;
      const noise = amp * Math.sin(Math.PI*t*3);
      pts.push([x + noise*dx/len, y + noise*dy/len]);
    }
    borderG.append("path")
      .attr("d","M"+pts.map(p=>p.join(",")).join("L"))
      .attr("class","fractal-border")
      .attr("stroke", "#58a6ff");
  });
}

// tooltip
const tooltip = document.getElementById("tooltip");
function showTooltip(e, html) {
  tooltip.innerHTML = html;
  tooltip.style.opacity = 1;
  tooltip.style.left = (e.offsetX+12)+"px";
  tooltip.style.top  = (e.offsetY-10)+"px";
}
function hideTooltip() { tooltip.style.opacity=0; }
</script>
</body>
</html>"""


def render_html(km: KnowledgeMap, output_path: str = "infom_graph.html") -> str:
    """Сгенерировать self-contained HTML файл с интерактивным графом."""
    data_json = _km_to_json(km)
    html      = HTML_TEMPLATE.replace("__DATA__", data_json)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
