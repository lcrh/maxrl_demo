// app.js -- Main application: wires controls, training loop, and rendering

import {
  MAZE_TRAIN,
  GridWorld,
  TabularSoftmaxPolicy,
  reinforceUpdate,
  maxrlUpdate,
  evaluateFromStart,
  bfsShortestPath,
} from "./engine.js";

import {
  Renderer,
  generateHeatmap,
  generateHeatmapMultistart,
  BG,
  DIVIDER,
  ACCENT_RL,
  ACCENT_MAXRL,
  CELL,
} from "./renderer.js";

import { MazeEditor } from "./editor.js";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CANVAS_W = 1400;
const CANVAS_H = 820;
const PANEL_W = 700;

// Layout Y positions (tuned to fit 11x11 grid + metrics + formulas + chart + HUD)
const GRID_Y = 8;
const GRID_MARGIN_X = 46; // center grid within each 700px panel

const CHART_H = 180;
const HUD_H = 44;

// Training defaults
const DEFAULTS = {
  multi: {
    lr: 0.3,
    N: 32,
    maxSteps: 25,
    evalInterval: 10,
    nEval: 64,
  },
  single: {
    lr: 0.3,
    N: 16,
    maxSteps: 80,
    evalInterval: 5,
    nEval: 64,
  },
};

const MAX_HISTORY = 2000;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const canvas = document.getElementById("mainCanvas");
const ctx = canvas.getContext("2d");
const renderer = new Renderer(CELL);
const editor = new MazeEditor(canvas, CELL);

// Grid / environment
let grid = MAZE_TRAIN.map((r) => [...r]);
let starts = [
  [1, 1],
  [9, 1],
];
let goal = [9, 9];
let env = new GridWorld(grid, starts[0], goal);
let startProbs = [0.5, 0.5];

// Mode
let multiStart = true;

// Policies
let rlPolicy = new TabularSoftmaxPolicy(env.height, env.width);
let maxrlPolicy = new TabularSoftmaxPolicy(env.height, env.width);

// Training state
let step = 0;
let paused = true;
let speed = 3; // steps-per-frame (mapped from slider)
let editorActive = false;

// History for charts
let history = _emptyHistory();

// Cached heatmap / path data (regenerated every evalInterval steps)
let rlHeat = null;
let maxrlHeat = null;
let rlMetrics = {};
let maxrlMetrics = {};

// ---------------------------------------------------------------------------
// History helpers
// ---------------------------------------------------------------------------

function _emptyHistory() {
  if (multiStart) {
    return {
      rl: { steps: [], start_0_p1: [], start_1_p1: [], K: [] },
      maxrl: { steps: [], start_0_p1: [], start_1_p1: [], K: [] },
    };
  }
  return {
    rl: { steps: [], pass_at_1: [], pass_at_k: [], entropy: [], unique_paths: [], K: [] },
    maxrl: { steps: [], pass_at_1: [], pass_at_k: [], entropy: [], unique_paths: [], K: [] },
  };
}

function _params() {
  return multiStart ? DEFAULTS.multi : DEFAULTS.single;
}

// ---------------------------------------------------------------------------
// Evaluation / heatmap refresh
// ---------------------------------------------------------------------------

function refreshEval() {
  const p = _params();

  if (multiStart) {
    // Heatmaps
    rlHeat = generateHeatmapMultistart(rlPolicy.logits, env, starts, 75, p.maxSteps);
    maxrlHeat = generateHeatmapMultistart(maxrlPolicy.logits, env, starts, 75, p.maxSteps);

    // Per-start pass@1
    const rlP = starts.map((s) => evaluateFromStart(rlPolicy, env, s, p.nEval, p.maxSteps));
    const mxP = starts.map((s) => evaluateFromStart(maxrlPolicy, env, s, p.nEval, p.maxSteps));

    rlMetrics = { start_0_p1: rlP[0], start_1_p1: rlP.length > 1 ? rlP[1] : rlP[0] };
    maxrlMetrics = { start_0_p1: mxP[0], start_1_p1: mxP.length > 1 ? mxP[1] : mxP[0] };
  } else {
    const start = starts[0];
    rlHeat = generateHeatmap(rlPolicy.logits, env, start, 150, p.maxSteps);
    maxrlHeat = generateHeatmap(maxrlPolicy.logits, env, start, 150, p.maxSteps);

    const rlP1 = evaluateFromStart(rlPolicy, env, start, p.nEval, p.maxSteps);
    const mxP1 = evaluateFromStart(maxrlPolicy, env, start, p.nEval, p.maxSteps);

    // pass@K estimate: 1 - (1-p1)^K
    const K = p.N;
    const rlPK = 1 - Math.pow(1 - rlP1, K);
    const mxPK = 1 - Math.pow(1 - mxP1, K);

    rlMetrics = { pass_at_1: rlP1, pass_at_k: rlPK, entropy: 0, unique_paths: 0, K };
    maxrlMetrics = { pass_at_1: mxP1, pass_at_k: mxPK, entropy: 0, unique_paths: 0, K };
  }
}

function recordHistory() {
  if (multiStart) {
    history.rl.steps.push(step);
    history.rl.start_0_p1.push(rlMetrics.start_0_p1);
    history.rl.start_1_p1.push(rlMetrics.start_1_p1);

    history.maxrl.steps.push(step);
    history.maxrl.start_0_p1.push(maxrlMetrics.start_0_p1);
    history.maxrl.start_1_p1.push(maxrlMetrics.start_1_p1);
  } else {
    history.rl.steps.push(step);
    history.rl.pass_at_1.push(rlMetrics.pass_at_1);
    history.rl.pass_at_k.push(rlMetrics.pass_at_k);

    history.maxrl.steps.push(step);
    history.maxrl.pass_at_1.push(maxrlMetrics.pass_at_1);
    history.maxrl.pass_at_k.push(maxrlMetrics.pass_at_k);
  }
}

// ---------------------------------------------------------------------------
// Reset training (keep grid/starts/goal)
// ---------------------------------------------------------------------------

function resetTraining() {
  rlPolicy = new TabularSoftmaxPolicy(env.height, env.width);
  maxrlPolicy = new TabularSoftmaxPolicy(env.height, env.width);
  step = 0;
  history = _emptyHistory();
  refreshEval();
  recordHistory();
}

function loadGrid(newGrid, newStarts, newGoal) {
  grid = newGrid.map((r) => [...r]);
  starts = newStarts.map((s) => [...s]);
  goal = [...newGoal];
  env = new GridWorld(grid, starts[0], goal);
  updateHardPct();  // set startProbs from slider
  resetTraining();
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

function render() {
  // Clear
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, CANVAS_W, CANVAS_H);

  const p = _params();
  const gridH = env.height;
  const gridW = env.width;
  const gridPxW = gridW * CELL;
  const gridPxH = gridH * CELL;

  // Center grids in each panel
  const rlOx = Math.floor((PANEL_W - gridPxW) / 2);
  const mxOx = PANEL_W + Math.floor((PANEL_W - gridPxW) / 2);
  const gy = GRID_Y;

  // Default heatmap if not yet generated
  const emptyHeat = new Float64Array(gridH * gridW);
  const rlV = rlHeat ? rlHeat.visits : emptyHeat;
  const mxV = maxrlHeat ? maxrlHeat.visits : emptyHeat;

  // Draw grids
  // starts[0]=(1,1)=hard, starts[1]=(9,1)=easy; extraStarts expects [easy, hard]
  const opts = multiStart ? { extraStarts: [starts.length > 1 ? starts[1] : starts[0], starts[0]] } : {};
  const rlGrid = renderer.drawGrid(ctx, env, rlV, null, rlOx, gy, "REINFORCE", ACCENT_RL, opts);
  const mxGrid = renderer.drawGrid(ctx, env, mxV, null, mxOx, gy, "MaxRL", ACCENT_MAXRL, opts);

  // Path traces
  if (multiStart) {
    if (rlHeat && rlHeat.pathsPerStart) {
      renderer.drawPathsMultistart(ctx, rlHeat.pathsPerStart, rlOx, rlGrid.oy, CELL);
    }
    if (maxrlHeat && maxrlHeat.pathsPerStart) {
      renderer.drawPathsMultistart(ctx, maxrlHeat.pathsPerStart, mxOx, mxGrid.oy, CELL);
    }
  } else {
    if (rlHeat && rlHeat.paths) {
      renderer.drawPathsSingle(ctx, rlHeat.paths, rlOx, rlGrid.oy, CELL);
    }
    if (maxrlHeat && maxrlHeat.paths) {
      renderer.drawPathsSingle(ctx, maxrlHeat.paths, mxOx, mxGrid.oy, CELL);
    }
  }

  // Metrics below grid
  const metricsY = rlGrid.oy + gridPxH + 16;
  if (multiStart) {
    // start_0=(1,1)=hard, start_1=(9,1)=easy; function expects (easy, hard)
    renderer.drawMetricsMultistart(ctx, rlMetrics.start_1_p1 || 0, rlMetrics.start_0_p1 || 0, rlOx, metricsY, gridPxW);
    renderer.drawMetricsMultistart(ctx, maxrlMetrics.start_1_p1 || 0, maxrlMetrics.start_0_p1 || 0, mxOx, metricsY, gridPxW);
  } else {
    renderer.drawMetricsSingle(ctx, rlMetrics, rlOx, metricsY, gridPxW);
    renderer.drawMetricsSingle(ctx, maxrlMetrics, mxOx, metricsY, gridPxW);
  }

  // Divider
  ctx.strokeStyle = DIVIDER;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(PANEL_W, 0);
  ctx.lineTo(PANEL_W, CANVAS_H - HUD_H);
  ctx.stroke();

  // Formulas
  const formulaY = metricsY + 36;
  const easyBFS = bfsShortestPath(grid, starts[starts.length - 1], goal);
  const hardBFS = multiStart && starts.length > 1
    ? bfsShortestPath(grid, starts[0], goal) : null;
  renderer.drawFormulas(ctx, rlOx, formulaY, {
    easyBFS: easyBFS ? easyBFS.distance : null,
    hardBFS: hardBFS ? hardBFS.distance : null,
    gridH: grid.length,
    gridW: grid[0].length,
    maxSteps: _params().maxSteps,
    multiStart,
  });

  // Chart
  const chartY = formulaY + 112;
  const chartW = CANVAS_W - 60;
  const chartRect = { x: 30, y: chartY, w: chartW, h: CHART_H };

  const histLen = history.rl.steps.length;
  if (multiStart) {
    renderer.drawChartMultistart(ctx, history, histLen - 1, chartRect);
  } else {
    renderer.drawChartSingle(ctx, history, histLen - 1, chartRect);
  }

  // HUD
  const hudY = CANVAS_H - HUD_H;
  const modeLabel = multiStart ? "Multi-start" : "Single-start";
  renderer.drawHUD(ctx, step, MAX_HISTORY * _params().evalInterval, paused, speed, modeLabel, {
    x: 0,
    y: hudY,
    w: CANVAS_W,
    h: HUD_H,
  });

  // Editor overlay (on top of everything)
  if (editorActive) {
    editor.draw();
  }
}

// ---------------------------------------------------------------------------
// Training loop (requestAnimationFrame)
// ---------------------------------------------------------------------------

let rafId = null;

function trainFrame() {
  rafId = requestAnimationFrame(trainFrame);

  if (paused || editorActive) {
    render();
    return;
  }

  const p = _params();

  for (let i = 0; i < speed; i++) {
    reinforceUpdate(rlPolicy, env, starts, startProbs, p.N, p.lr, p.maxSteps);
    maxrlUpdate(maxrlPolicy, env, starts, startProbs, p.N, p.lr, p.maxSteps);
    step++;

    if (step % p.evalInterval === 0) {
      refreshEval();
      recordHistory();
    }
  }

  // If we haven't evaluated this frame (speed < evalInterval), still update visuals periodically
  if (step % p.evalInterval !== 0 && (!rlHeat || !maxrlHeat)) {
    refreshEval();
  }

  render();
}

// ---------------------------------------------------------------------------
// Speed mapping: slider 1-100 -> steps per frame 1-20
// ---------------------------------------------------------------------------

function sliderToSpeed(val) {
  return Math.max(1, Math.round((val / 100) * 20));
}

function speedToSlider(spd) {
  return Math.round((spd / 20) * 100);
}

// ---------------------------------------------------------------------------
// Button state helpers
// ---------------------------------------------------------------------------

const btnDefaultGrid = document.getElementById("btnDefaultGrid");
const btnRandomGrid = document.getElementById("btnRandomGrid");
const btnEditGrid = document.getElementById("btnEditGrid");
const btnStartMode = document.getElementById("btnStartMode");
const btnPlayPause = document.getElementById("btnPlayPause");
const btnReset = document.getElementById("btnReset");
const speedSlider = document.getElementById("speedSlider");
const hardPctSlider = document.getElementById("hardPctSlider");
const hardPctLabel = document.getElementById("hardPctLabel");

function updateHardPct() {
  const pct = Number(hardPctSlider.value);
  hardPctLabel.textContent = `Hard: ${pct}%`;
  if (multiStart && starts.length > 1) {
    const hardFrac = pct / 100;
    startProbs = [hardFrac, 1 - hardFrac];  // starts[0]=hard, starts[1]=easy
  }
}

function updateGridButtons(active) {
  btnDefaultGrid.classList.toggle("active", active === "default");
  btnRandomGrid.classList.toggle("active", active === "random");
  btnEditGrid.classList.toggle("active", active === "edit");
}

function updatePlayPauseButton() {
  if (paused) {
    btnPlayPause.textContent = "Play";
    btnPlayPause.classList.add("accent-orange");
  } else {
    btnPlayPause.textContent = "Pause";
    btnPlayPause.classList.remove("accent-orange");
  }
}

function updateModeButton() {
  if (multiStart) {
    btnStartMode.textContent = "Multi Start";
    btnStartMode.classList.add("on");
    hardPctSlider.style.display = "";
    hardPctLabel.style.display = "";
  } else {
    btnStartMode.textContent = "Single Start";
    btnStartMode.classList.remove("on");
    hardPctSlider.style.display = "none";
    hardPctLabel.style.display = "none";
  }
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

function togglePlayPause() {
  paused = !paused;
  updatePlayPauseButton();
}

function doReset() {
  resetTraining();
  render();
}

function toggleMode() {
  multiStart = !multiStart;
  updateModeButton();
  if (!multiStart) {
    startProbs = [1.0];
  } else {
    updateHardPct();
  }
  resetTraining();
  render();
}

function loadDefaultGrid() {
  if (editorActive) closeEditor();
  updateGridButtons("default");
  editor.loadDefault();
  const st = editor.getState();
  // Multi-start default: easy=(1,1), hard=(9,1)
  const allStarts = multiStart ? [st.hardStart, st.starts[0]] : [st.starts[0]];
  loadGrid(st.grid, allStarts, st.goal);
  render();
}

function loadRandomGrid() {
  if (editorActive) closeEditor();
  updateGridButtons("random");
  editor.randomize(11, 11);
  const st = editor.getState();
  const allStarts = multiStart ? [st.hardStart, st.starts[0]] : [st.starts[0]];
  loadGrid(st.grid, allStarts, st.goal);
  render();
}

function openEditor() {
  editorActive = true;
  paused = true;
  updatePlayPauseButton();
  updateGridButtons("edit");
  btnEditGrid.textContent = "Done Editing";
  editor.open();
}

function closeEditor() {
  editor.close();
  btnEditGrid.textContent = "Edit Grid";
  editorActive = false;
  updateGridButtons("default");

  const st = editor.getState();
  if (!st.valid) {
    // Invalid maze -- show warning, don't load
    renderer.drawLoading(ctx, "Invalid maze: some starts cannot reach goal!", CANVAS_W, CANVAS_H);
    return;
  }

  const allStarts = multiStart ? [st.hardStart, st.starts[0]] : [st.starts[0]];
  loadGrid(st.grid, allStarts, st.goal);
}

function toggleEditor() {
  if (editorActive) {
    closeEditor();
    render();
  } else {
    openEditor();
  }
}

function changeSpeed(delta) {
  speed = Math.max(1, Math.min(20, speed + delta));
  speedSlider.value = speedToSlider(speed);
}

// -- Button listeners -------------------------------------------------------

btnPlayPause.addEventListener("click", togglePlayPause);
btnReset.addEventListener("click", doReset);
btnDefaultGrid.addEventListener("click", loadDefaultGrid);
btnRandomGrid.addEventListener("click", loadRandomGrid);
btnEditGrid.addEventListener("click", toggleEditor);
btnStartMode.addEventListener("click", toggleMode);
speedSlider.addEventListener("input", () => {
  speed = sliderToSpeed(Number(speedSlider.value));
});
hardPctSlider.addEventListener("input", updateHardPct);

// -- Keyboard shortcuts -------------------------------------------------------

document.addEventListener("keydown", (e) => {
  // Don't capture if user is typing in an input
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  switch (e.code) {
    case "Space":
      e.preventDefault();
      togglePlayPause();
      break;
    case "KeyR":
      doReset();
      break;
    case "KeyM":
      toggleMode();
      break;
    case "KeyF":
      changeSpeed(1);
      break;
    case "KeyS":
      changeSpeed(-1);
      break;
    case "KeyE":
      toggleEditor();
      break;
    case "Escape":
      if (editorActive) {
        closeEditor();
        render();
      }
      break;
  }
});

// -- Visibility change: pause when tab hidden ---------------------------------

document.addEventListener("visibilitychange", () => {
  if (document.hidden && !paused) {
    paused = true;
    updatePlayPauseButton();
  }
});

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

// Set initial slider
speedSlider.value = speedToSlider(speed);

// Set initial button states
updateGridButtons("default");
updatePlayPauseButton();
updateModeButton();

// Initial eval so we have something to render
refreshEval();
recordHistory();

// Start the render/training loop
render();
trainFrame();
