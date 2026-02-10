// renderer.js -- Canvas 2D renderer (port of pygame Renderer from src/main.py)

import { rollout, TabularSoftmaxPolicy } from "./engine.js";

// ---------------------------------------------------------------------------
// Color constants (match pygame exactly)
// ---------------------------------------------------------------------------
export const BG           = "#121218";
export const WALL         = "#28282e";
export const PATH_COL     = "#16161e";
export const GRID_LINE    = "#32323c";
export const START_COL    = "#00c850";
export const START_HARD   = "#dcb400";
export const GOAL_COL     = "#dc2828";
export const TEXT_COL     = "#dcdce6";
export const DIM_TEXT     = "#8c8ca0";
export const ACCENT_RL    = "#64a0ff";
export const ACCENT_MAXRL = "#ff8c3c";
export const CHART_BG     = "#1c1c26";
export const DIVIDER      = "#3c3c4b";

// Heat gradient stops: [r, g, b, a]  (a is 0-255)
const HEAT_COLORS = [
  [0,   0,   0,   0],
  [80,  40,  0,   60],
  [160, 80,  0,   110],
  [220, 140, 0,   170],
  [255, 200, 40,  220],
  [255, 255, 100, 255],
];

// Trace colors: [r, g, b]
const TRACE_EASY   = [[0, 220, 120], [60, 200, 100], [100, 240, 140]];
const TRACE_HARD   = [[200, 80, 255], [255, 100, 200], [180, 120, 255]];
const TRACE_SINGLE = [
  [0, 220, 120], [60, 160, 255], [255, 100, 200],
  [255, 200, 40], [140, 100, 255],
];

// Default cell size (slightly smaller than pygame's 40px to fit web)
export const CELL = 36;

// ---------------------------------------------------------------------------
// Color helpers
// ---------------------------------------------------------------------------

function _lerp(c1, c2, t) {
  const out = [];
  for (let i = 0; i < c1.length; i++) {
    out.push(c1[i] + (c2[i] - c1[i]) * t);
  }
  return out;
}

export function heatColor(val) {
  val = Math.max(0, Math.min(1, val));
  const n = HEAT_COLORS.length - 1;
  const idx = val * n;
  const lo = Math.floor(idx);
  const hi = Math.min(lo + 1, n);
  const c = _lerp(HEAT_COLORS[lo], HEAT_COLORS[hi], idx - lo);
  return `rgba(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])},${(c[3] / 255).toFixed(3)})`;
}

function _rgb(arr) {
  return `rgb(${arr[0]},${arr[1]},${arr[2]})`;
}

// ---------------------------------------------------------------------------
// Data generation helpers
// ---------------------------------------------------------------------------

/**
 * Generate heatmap from rollouts (single-start or from a specific start).
 * Returns flat Float64Array of visits (H*W, normalized 0-1) and successful paths.
 */
export function generateHeatmap(logits, env, start, nRollouts = 150, maxSteps = 80) {
  const policy = new TabularSoftmaxPolicy(env.height, env.width);
  policy.logits.set(logits);
  const visits = new Float64Array(env.height * env.width);
  const paths = [];
  for (let i = 0; i < nRollouts; i++) {
    const traj = rollout(policy, env, start, maxSteps);
    for (const [r, c] of traj.path) {
      visits[r * env.width + c] += 1;
    }
    if (traj.reachedGoal) {
      paths.push(traj.path);
    }
  }
  let mx = 0;
  for (let i = 0; i < visits.length; i++) {
    if (visits[i] > mx) mx = visits[i];
  }
  if (mx > 0) {
    for (let i = 0; i < visits.length; i++) visits[i] /= mx;
  }
  return { visits, paths };
}

/**
 * Multi-start heatmap: rollout from each start, return combined heatmap and per-start paths.
 */
export function generateHeatmapMultistart(logits, env, starts, nPerStart = 75, maxSteps = 25) {
  const policy = new TabularSoftmaxPolicy(env.height, env.width);
  policy.logits.set(logits);
  const visits = new Float64Array(env.height * env.width);
  const pathsPerStart = starts.map(() => []);
  for (let si = 0; si < starts.length; si++) {
    const start = starts[si];
    for (let i = 0; i < nPerStart; i++) {
      const traj = rollout(policy, env, start, maxSteps);
      for (const [r, c] of traj.path) {
        visits[r * env.width + c] += 1;
      }
      if (traj.reachedGoal) {
        pathsPerStart[si].push(traj.path);
      }
    }
  }
  let mx = 0;
  for (let i = 0; i < visits.length; i++) {
    if (visits[i] > mx) mx = visits[i];
  }
  if (mx > 0) {
    for (let i = 0; i < visits.length; i++) visits[i] /= mx;
  }
  return { visits, pathsPerStart };
}

// ---------------------------------------------------------------------------
// Renderer class
// ---------------------------------------------------------------------------

export class Renderer {
  constructor(cellSize = CELL) {
    this.cell = cellSize;
  }

  // -- Loading screen -------------------------------------------------------

  drawLoading(ctx, msg, width, height) {
    ctx.fillStyle = BG;
    ctx.fillRect(0, 0, width, height);
    ctx.fillStyle = TEXT_COL;
    ctx.font = "bold 28px -apple-system, 'Helvetica Neue', Arial, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(msg, width / 2, height / 2);
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  }

  // -- Grid panel -----------------------------------------------------------

  /**
   * Draw the maze grid with heatmap overlay, grid lines, start/goal markers.
   * @param {CanvasRenderingContext2D} ctx
   * @param {object} env - GridWorld
   * @param {Float64Array} heatmap - flat visits array (H*W), normalized 0-1
   * @param {number[][]} paths - unused here (drawn separately via drawPaths*)
   * @param {number} ox - left x offset
   * @param {number} oy - top y offset
   * @param {string} label - title text
   * @param {string} accentColor - CSS color for label
   * @param {object} options - { extraStarts: [[r,c],[r,c]] } for multi-start mode
   * @returns {{ oy: number, gw: number, gh: number }} adjusted origin y and grid dims
   */
  drawGrid(ctx, env, heatmap, paths, ox, oy, label, accentColor, options = {}) {
    const C = this.cell;
    const H = env.height;
    const W = env.width;
    const gw = W * C;
    const gh = H * C;

    // Label
    ctx.font = "bold 20px -apple-system, 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = accentColor;
    ctx.textAlign = "center";
    ctx.fillText(label, ox + gw / 2, oy + 18);
    ctx.textAlign = "start";
    oy += 30;

    // Border
    ctx.strokeStyle = GRID_LINE;
    ctx.lineWidth = 2;
    _roundRect(ctx, ox - 2, oy - 2, gw + 4, gh + 4, 4);
    ctx.stroke();

    // Cells + heatmap overlay
    for (let r = 0; r < H; r++) {
      for (let c = 0; c < W; c++) {
        const rx = ox + c * C;
        const ry = oy + r * C;
        if (env.grid[r][c] === 1) {
          ctx.fillStyle = WALL;
          ctx.fillRect(rx, ry, C, C);
        } else {
          ctx.fillStyle = PATH_COL;
          ctx.fillRect(rx, ry, C, C);
          const v = heatmap[r * W + c];
          if (v > 0.01) {
            ctx.fillStyle = heatColor(v);
            ctx.fillRect(rx, ry, C, C);
          }
        }
        // Grid line
        ctx.strokeStyle = GRID_LINE;
        ctx.lineWidth = 1;
        ctx.strokeRect(rx, ry, C, C);
      }
    }

    // Start / Goal markers
    const markers = [];
    markers.push({ pos: env.goal, color: GOAL_COL, text: "G" });

    if (options.extraStarts) {
      // multi-start: extraStarts = [easy, hard]
      markers.push({ pos: options.extraStarts[0], color: START_COL, text: "E" });
      markers.push({ pos: options.extraStarts[1], color: START_HARD, text: "H" });
    } else {
      markers.push({ pos: env.start, color: START_COL, text: "S" });
    }

    for (const m of markers) {
      const [pr, pc] = m.pos;
      const mx = ox + pc * C + 4;
      const my = oy + pr * C + 4;
      const mw = C - 8;
      const mh = C - 8;
      ctx.fillStyle = m.color;
      _roundRect(ctx, mx, my, mw, mh, 6);
      ctx.fill();
      ctx.fillStyle = "#ffffff";
      ctx.font = "bold 13px -apple-system, 'Helvetica Neue', Arial, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(m.text, mx + mw / 2, my + mh / 2);
    }

    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";

    return { oy, gw, gh };
  }

  // -- Path traces ----------------------------------------------------------

  drawPathsSingle(ctx, paths, ox, oy, cellSize) {
    const C = cellSize || this.cell;
    if (!paths || paths.length === 0) return;

    // Deduplicate paths, keep up to 5
    const seen = new Set();
    const unique = [];
    for (const p of paths) {
      const key = p.map(([r, c]) => `${r},${c}`).join("|");
      if (!seen.has(key)) {
        seen.add(key);
        unique.push(p);
      }
      if (unique.length >= 5) break;
    }

    for (let i = 0; i < unique.length; i++) {
      const path = unique[i];
      const col = TRACE_SINGLE[i % TRACE_SINGLE.length];
      ctx.save();
      ctx.globalAlpha = 160 / 255;
      ctx.strokeStyle = _rgb(col);
      ctx.lineWidth = 3;
      ctx.lineJoin = "round";
      ctx.lineCap = "round";
      ctx.beginPath();
      for (let j = 0; j < path.length; j++) {
        const [r, c] = path[j];
        const x = ox + c * C + C / 2;
        const y = oy + r * C + C / 2;
        if (j === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.restore();
    }
  }

  drawPathsMultistart(ctx, pathsPerStart, ox, oy, cellSize) {
    const C = cellSize || this.cell;
    const palettes = [TRACE_HARD, TRACE_EASY]; // index 0=hard, 1=easy (matches pygame)

    for (let si = 0; si < pathsPerStart.length; si++) {
      const paths = pathsPerStart[si];
      if (!paths || paths.length === 0) continue;
      const palette = palettes[Math.min(si, palettes.length - 1)];

      // Deduplicate, keep up to 3 per start
      const seen = new Set();
      const unique = [];
      for (const p of paths) {
        const key = p.map(([r, c]) => `${r},${c}`).join("|");
        if (!seen.has(key)) {
          seen.add(key);
          unique.push(p);
        }
        if (unique.length >= 3) break;
      }

      for (let i = 0; i < unique.length; i++) {
        const path = unique[i];
        const col = palette[i % palette.length];
        ctx.save();
        ctx.globalAlpha = 180 / 255;
        ctx.strokeStyle = _rgb(col);
        ctx.lineWidth = 3;
        ctx.lineJoin = "round";
        ctx.lineCap = "round";
        ctx.beginPath();
        for (let j = 0; j < path.length; j++) {
          const [r, c] = path[j];
          const x = ox + c * C + C / 2;
          const y = oy + r * C + C / 2;
          if (j === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();
      }
    }
  }

  // -- Metrics text ---------------------------------------------------------

  drawMetricsSingle(ctx, metrics, ox, oy, gw) {
    const p1 = metrics.pass_at_1 ?? 0;
    const pk = metrics.pass_at_k ?? 0;
    const ent = metrics.entropy ?? 0;
    const up = metrics.unique_paths ?? 0;
    const half = gw / 2;

    ctx.font = "16px -apple-system, 'Helvetica Neue', Arial, sans-serif";

    ctx.fillStyle = TEXT_COL;
    ctx.fillText(`Pass@1:  ${p1.toFixed(2)}`, ox, oy);
    ctx.fillText(`Pass@16: ${pk.toFixed(2)}`, ox + half, oy);

    ctx.fillStyle = DIM_TEXT;
    ctx.fillText(`Entropy: ${ent.toFixed(2)}`, ox, oy + 22);
    ctx.fillText(`Paths:   ${up}`, ox + half, oy + 22);
  }

  drawMetricsMultistart(ctx, easyP1, hardP1, ox, oy, gw) {
    const half = gw / 2;
    ctx.font = "16px -apple-system, 'Helvetica Neue', Arial, sans-serif";

    ctx.fillStyle = START_COL;
    ctx.fillText(`Easy start: ${(easyP1 * 100).toFixed(0)}%`, ox, oy);

    ctx.fillStyle = START_HARD;
    ctx.fillText(`Hard start: ${(hardP1 * 100).toFixed(0)}%`, ox + half, oy);
  }

  // -- Formulas -------------------------------------------------------------

  drawFormulas(ctx, ox, oy, info = {}) {
    const { easyBFS, hardBFS, gridH, gridW, maxSteps, multiStart } = info;

    ctx.font = "13px -apple-system, 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = DIM_TEXT;
    ctx.fillText(
      "Tajwar et al. 2025 -- Maximum Likelihood Reinforcement Learning",
      ox, oy
    );

    ctx.font = "13px 'Courier New', Courier, monospace";
    ctx.fillStyle = ACCENT_RL;
    ctx.fillText("RL:    max  E_maze[  p(success|maze)  ]", ox, oy + 20);

    ctx.fillStyle = ACCENT_MAXRL;
    ctx.fillText("MaxRL: max  E_maze[ log p(success|maze) ]", ox, oy + 38);

    ctx.font = "13px -apple-system, 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = TEXT_COL;
    ctx.fillText(
      "RL averages p across mazes -- hard mazes contribute little gradient." +
      "  MaxRL averages log p -- hard mazes matter equally.",
      ox, oy + 58
    );

    // Maze info line
    const parts = [];
    parts.push(`${gridH}\u00d7${gridW} grid`);
    parts.push("obs = (row, col)");
    parts.push("actions = {up, down, left, right}");
    parts.push("tabular softmax policy");
    if (maxSteps != null) parts.push(`budget = ${maxSteps} steps`);
    ctx.fillStyle = DIM_TEXT;
    ctx.fillText(parts.join("  |  "), ox, oy + 78);

    // BFS distances
    if (multiStart && easyBFS != null && hardBFS != null) {
      ctx.fillStyle = START_COL;
      ctx.fillText(`Easy start: BFS = ${easyBFS} steps`, ox, oy + 96);
      ctx.fillStyle = START_HARD;
      ctx.fillText(`Hard start: BFS = ${hardBFS} steps`, ox + 220, oy + 96);
    } else if (easyBFS != null) {
      ctx.fillStyle = DIM_TEXT;
      ctx.fillText(`Shortest path: BFS = ${easyBFS} steps`, ox, oy + 96);
    }
  }

  // -- Chart shared helpers -------------------------------------------------

  _drawChartFrame(ctx, rect) {
    const { x, y, w, h } = rect;

    // Background
    ctx.fillStyle = CHART_BG;
    _roundRect(ctx, x, y, w, h, 8);
    ctx.fill();

    // Border
    ctx.strokeStyle = DIVIDER;
    ctx.lineWidth = 1;
    _roundRect(ctx, x, y, w, h, 8);
    ctx.stroke();

    const pl = 70, pr = 30, pt = 30, pb = 36;
    const px = x + pl;
    const py = y + pt;
    const pw = w - pl - pr;
    const ph = h - pt - pb;

    // Axes
    ctx.strokeStyle = DIVIDER;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(px, py + ph);
    ctx.lineTo(px + pw, py + ph);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(px, py);
    ctx.lineTo(px, py + ph);
    ctx.stroke();

    // Y-axis labels and grid lines
    ctx.font = "11px -apple-system, 'Helvetica Neue', Arial, sans-serif";
    for (const val of [0, 0.25, 0.5, 0.75, 1.0]) {
      const yy = py + ph - Math.round(val * ph);

      ctx.fillStyle = DIM_TEXT;
      ctx.textAlign = "right";
      ctx.fillText(`${(val * 100).toFixed(0)}%`, px - 8, yy + 4);

      ctx.strokeStyle = "#28283a";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, yy);
      ctx.lineTo(px + pw, yy);
      ctx.stroke();
    }
    ctx.textAlign = "start";

    return { px, py, pw, ph };
  }

  _plotLine(ctx, data, color, npts, px, py, pw, ph, total, dashed = false) {
    if (npts < 2) return;
    const pts = [];
    for (let i = 0; i < npts; i++) {
      const x = px + Math.round(i / total * pw);
      const y = py + ph - Math.round(Math.max(0, Math.min(1, data[i])) * ph);
      pts.push([x, y]);
    }

    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    if (dashed) {
      ctx.setLineDash([12, 8]);
    } else {
      ctx.setLineDash([]);
    }

    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      if (i === 0) ctx.moveTo(pts[i][0], pts[i][1]);
      else ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.stroke();
    ctx.restore();
  }

  _drawLegend(ctx, items, px, py, pw) {
    const legendW = 170;
    const legendH = items.length * 18 + 8;
    const legendX = px + pw - legendW - 8;
    const legendY = py + 6;

    // Background
    ctx.fillStyle = CHART_BG;
    _roundRect(ctx, legendX - 6, legendY - 4, legendW + 12, legendH + 4, 4);
    ctx.fill();
    ctx.strokeStyle = DIVIDER;
    ctx.lineWidth = 1;
    _roundRect(ctx, legendX - 6, legendY - 4, legendW + 12, legendH + 4, 4);
    ctx.stroke();

    ctx.lineWidth = 3;
    ctx.lineCap = "round";

    for (let i = 0; i < items.length; i++) {
      const [color, text, isDashed] = items[i];
      const ly = legendY + i * 18 + 2;

      ctx.strokeStyle = color;
      if (isDashed) {
        // Two short dashes
        ctx.beginPath();
        ctx.moveTo(legendX, ly + 6);
        ctx.lineTo(legendX + 9, ly + 6);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(legendX + 14, ly + 6);
        ctx.lineTo(legendX + 23, ly + 6);
        ctx.stroke();
      } else {
        ctx.beginPath();
        ctx.moveTo(legendX, ly + 6);
        ctx.lineTo(legendX + 23, ly + 6);
        ctx.stroke();
      }

      ctx.font = "13px -apple-system, 'Helvetica Neue', Arial, sans-serif";
      ctx.fillStyle = color;
      ctx.fillText(text, legendX + 28, ly + 10);
    }
  }

  // -- Chart: single-start mode ---------------------------------------------

  drawChartSingle(ctx, history, currentIdx, rect) {
    const { px, py, pw, ph } = this._drawChartFrame(ctx, rect);
    if (currentIdx < 1) return;
    const n = currentIdx + 1;
    const total = Math.max(1, history.rl.pass_at_1.length - 1);

    const rl = history.rl;
    const mx = history.maxrl;

    this._plotLine(ctx, rl.pass_at_1,  ACCENT_RL,    n, px, py, pw, ph, total);
    this._plotLine(ctx, rl.pass_at_k,  ACCENT_RL,    n, px, py, pw, ph, total, true);
    this._plotLine(ctx, mx.pass_at_1,  ACCENT_MAXRL, n, px, py, pw, ph, total);
    this._plotLine(ctx, mx.pass_at_k,  ACCENT_MAXRL, n, px, py, pw, ph, total, true);

    this._drawLegend(ctx, [
      [ACCENT_RL,    "RL pass@1",    false],
      [ACCENT_RL,    "RL pass@16",   true],
      [ACCENT_MAXRL, "MaxRL pass@1", false],
      [ACCENT_MAXRL, "MaxRL pass@16", true],
    ], px, py, pw);

    // Training step label
    const steps = rl.steps;
    const s = steps[Math.min(n - 1, steps.length - 1)];
    ctx.font = "11px -apple-system, 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = DIM_TEXT;
    ctx.textAlign = "right";
    ctx.fillText(`Training step ${s}`, px + pw, py + ph + 18);
    ctx.textAlign = "start";
  }

  // -- Chart: multi-start mode ----------------------------------------------

  drawChartMultistart(ctx, history, currentIdx, rect) {
    const { px, py, pw, ph } = this._drawChartFrame(ctx, rect);
    if (currentIdx < 1) return;
    const n = currentIdx + 1;
    const total = Math.max(1, history.rl.start_0_p1.length - 1);

    const rl = history.rl;
    const mx = history.maxrl;

    // Easy start (solid), Hard start (dashed)
    this._plotLine(ctx, rl.start_1_p1, ACCENT_RL,    n, px, py, pw, ph, total);
    this._plotLine(ctx, rl.start_0_p1, ACCENT_RL,    n, px, py, pw, ph, total, true);
    this._plotLine(ctx, mx.start_1_p1, ACCENT_MAXRL, n, px, py, pw, ph, total);
    this._plotLine(ctx, mx.start_0_p1, ACCENT_MAXRL, n, px, py, pw, ph, total, true);

    this._drawLegend(ctx, [
      [ACCENT_RL,    "RL easy start",    false],
      [ACCENT_RL,    "RL hard start",    true],
      [ACCENT_MAXRL, "MaxRL easy start", false],
      [ACCENT_MAXRL, "MaxRL hard start", true],
    ], px, py, pw);

    // Training step label
    const steps = rl.steps;
    const s = steps[Math.min(n - 1, steps.length - 1)];
    ctx.font = "11px -apple-system, 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = DIM_TEXT;
    ctx.textAlign = "right";
    ctx.fillText(`Training step ${s}`, px + pw, py + ph + 18);
    ctx.textAlign = "start";
  }

  // -- HUD (bottom bar) -----------------------------------------------------

  drawHUD(ctx, step, total, paused, speed, modeLabel, rect) {
    const { x, y, w, h } = rect;

    // Background
    ctx.fillStyle = "#1e1e28";
    ctx.fillRect(x, y, w, h);

    // Top divider line
    ctx.strokeStyle = DIVIDER;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + w, y);
    ctx.stroke();

    const xStart = x + 20;
    const cy = y + h / 2;

    // Step counter
    ctx.font = "16px -apple-system, 'Helvetica Neue', Arial, sans-serif";
    ctx.fillStyle = TEXT_COL;
    ctx.textBaseline = "middle";
    ctx.fillText(`Step ${step}/${total}`, xStart, cy);

    // Progress bar
    const bx = xStart + 140;
    const bw = 200;
    const bh = 12;
    const by = cy - bh / 2;

    ctx.fillStyle = "#32323c";
    _roundRect(ctx, bx, by, bw, bh, 6);
    ctx.fill();

    const fill = Math.round(bw * step / Math.max(1, total));
    if (fill > 0) {
      ctx.fillStyle = ACCENT_MAXRL;
      _roundRect(ctx, bx, by, fill, bh, 6);
      ctx.fill();
    }

    // Controls hints
    let cx = bx + bw + 40;
    const controls = [
      ["[SPACE]", paused ? "Play" : "Pause"],
      ["[R]", "Reset"],
      ["[M]", modeLabel],
      ["[F/S]", `Speed ${speed}x`],
    ];

    for (const [key, action] of controls) {
      ctx.font = "bold 13px -apple-system, 'Helvetica Neue', Arial, sans-serif";
      ctx.fillStyle = ACCENT_MAXRL;
      ctx.fillText(key, cx, cy);
      const kw = ctx.measureText(key).width;

      ctx.font = "13px -apple-system, 'Helvetica Neue', Arial, sans-serif";
      ctx.fillStyle = DIM_TEXT;
      ctx.fillText(` ${action}`, cx + kw, cy);
      const aw = ctx.measureText(` ${action}`).width;

      cx += kw + aw + 20;
    }

    ctx.textBaseline = "alphabetic";
  }
}

// ---------------------------------------------------------------------------
// Canvas helper: rounded rectangle path
// ---------------------------------------------------------------------------

function _roundRect(ctx, x, y, w, h, r) {
  if (w <= 0 || h <= 0) return;
  r = Math.min(r, w / 2, h / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h - r);
  ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
  ctx.lineTo(x + r, y + h);
  ctx.arcTo(x, y + h, x, y + h - r, r);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.closePath();
}
