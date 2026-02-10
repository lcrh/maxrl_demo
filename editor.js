// editor.js -- Interactive grid editor & maze generation UI
import { MAZE_TRAIN, bfsShortestPath, generateMaze } from './engine.js';

// ── Colours (match style.css palette) ────────────────────────────────────────
const COL_WALL      = '#1a1a24';
const COL_PATH      = '#3c3c52';
const COL_BORDER    = '#2a2a36';
const COL_HOVER     = 'rgba(255,255,255,0.18)';
const COL_START     = '#00c850';   // easy start (green)
const COL_HARD      = '#f0c030';   // hard start (yellow)
const COL_GOAL      = '#ff4444';   // goal (red)
const COL_INVALID   = '#ff2222';
const COL_OVERLAY   = 'rgba(0,0,0,0.55)';
const COL_GRID_LINE = 'rgba(255,255,255,0.06)';
const COL_TEXT_DIM  = '#8c8c9e';

// ── Default start / goal for the MAZE_TRAIN layout ──────────────────────────
const DEFAULT_STARTS = [[1, 1]];
const DEFAULT_HARD   = [9, 1];
const DEFAULT_GOAL   = [9, 9];

// ─────────────────────────────────────────────────────────────────────────────
// MazeEditor
// ─────────────────────────────────────────────────────────────────────────────

export class MazeEditor {
  /**
   * @param {HTMLCanvasElement} canvas
   * @param {number} cellSize - pixel size of each cell in the editor overlay
   */
  constructor(canvas, cellSize = 36) {
    this.canvas  = canvas;
    this.ctx     = canvas.getContext('2d');
    this.cellPx  = cellSize;

    // State
    this.grid    = deepCopyGrid(MAZE_TRAIN);
    this.starts  = DEFAULT_STARTS.map(s => [...s]);
    this.hardStart = [...DEFAULT_HARD];
    this.goal    = [...DEFAULT_GOAL];
    this.active  = false;     // editor overlay visible?
    this.valid   = true;      // all starts reachable?

    // Interaction
    this.hoverCell = null;    // [r, c] under cursor
    this.painting  = false;   // dragging?
    this.paintVal  = null;    // 0 or 1 while dragging

    // Derived layout (set in _layout)
    this.offsetX = 0;
    this.offsetY = 0;
    this._layout();

    // Bind handlers (stored so we can remove them)
    this._onMouseMove  = this._handleMouseMove.bind(this);
    this._onMouseDown  = this._handleMouseDown.bind(this);
    this._onMouseUp    = this._handleMouseUp.bind(this);
    this._onMouseLeave = this._handleMouseLeave.bind(this);
  }

  // ── Public API ────────────────────────────────────────────────────────────

  /** Activate editor overlay and attach mouse listeners. */
  open() {
    this.active = true;
    this._layout();
    this.canvas.addEventListener('mousemove',  this._onMouseMove);
    this.canvas.addEventListener('mousedown',  this._onMouseDown);
    this.canvas.addEventListener('mouseup',    this._onMouseUp);
    this.canvas.addEventListener('mouseleave', this._onMouseLeave);
    this.draw();
  }

  /** Deactivate editor and remove listeners. */
  close() {
    this.active = false;
    this.hoverCell = null;
    this.painting  = false;
    this.canvas.removeEventListener('mousemove',  this._onMouseMove);
    this.canvas.removeEventListener('mousedown',  this._onMouseDown);
    this.canvas.removeEventListener('mouseup',    this._onMouseUp);
    this.canvas.removeEventListener('mouseleave', this._onMouseLeave);
  }

  /** Load the default MAZE_TRAIN grid and markers. */
  loadDefault() {
    this.grid      = deepCopyGrid(MAZE_TRAIN);
    this.starts    = DEFAULT_STARTS.map(s => [...s]);
    this.hardStart = [...DEFAULT_HARD];
    this.goal      = [...DEFAULT_GOAL];
    this._layout();
    this._validate();
    if (this.active) this.draw();
  }

  /** Generate a random maze. */
  randomize(rows = 11, cols = 11) {
    // Try up to 20 mazes to find one with good easy/hard separation
    const maxAttempts = 20;
    const maxPath = rows + cols;  // neither start should be absurdly far

    for (let attempt = 0; attempt < maxAttempts; attempt++) {
      this.grid = generateMaze(rows, cols);
      const h = this.grid.length;
      const w = this.grid[0].length;
      this.goal = [h - 2, w - 2];

      // Compute BFS distance for every open cell to goal
      const cells = [];
      for (let r = 1; r < h - 1; r++) {
        for (let c = 1; c < w - 1; c++) {
          if (this.grid[r][c] === 0) {
            const res = bfsShortestPath(this.grid, [r, c], this.goal);
            if (res) cells.push([r, c, res.distance]);
          }
        }
      }
      if (cells.length < 2) continue;

      cells.sort((a, b) => a[2] - b[2]);

      // Easy start: pick from the closest third (but not distance 0 = goal)
      const easyPool = cells.filter(c => c[2] > 0 && c[2] <= maxPath);
      const hardPool = cells.filter(c => c[2] > 0 && c[2] <= maxPath);
      if (easyPool.length < 2) continue;

      // Pick easy from bottom quarter, hard from top quarter of distances
      const q1 = Math.max(1, Math.floor(easyPool.length * 0.25));
      const q3 = Math.floor(hardPool.length * 0.75);

      const easy = easyPool[Math.floor(Math.random() * q1)];
      const hard = hardPool[q3 + Math.floor(Math.random() * (hardPool.length - q3))];

      // Verify: easy shorter than hard, and hard is at least 1.5x easy
      if (easy[2] < hard[2] && hard[2] >= easy[2] * 1.5) {
        this.starts = [[easy[0], easy[1]]];
        this.hardStart = [hard[0], hard[1]];
        this._layout();
        this._validate();
        if (this.active) this.draw();
        return;
      }
    }

    // Fallback: use whatever we got from the last attempt
    const h = this.grid.length;
    const w = this.grid[0].length;
    this.goal = [h - 2, w - 2];
    this.starts = [[h - 2, 1]];
    this.hardStart = [1, 1];

    this._layout();
    this._validate();
    if (this.active) this.draw();
  }

  /** Return current editor state for the training loop. */
  getState() {
    return {
      grid:      deepCopyGrid(this.grid),
      starts:    this.starts.map(s => [...s]),
      hardStart: [...this.hardStart],
      goal:      [...this.goal],
      valid:     this.valid,
    };
  }

  // ── Drawing ───────────────────────────────────────────────────────────────

  /** Full editor overlay render. */
  draw() {
    if (!this.active) return;
    const ctx = this.ctx;
    const cw  = this.canvas.width;
    const ch  = this.canvas.height;

    // Dim background
    ctx.fillStyle = COL_OVERLAY;
    ctx.fillRect(0, 0, cw, ch);

    const rows = this.grid.length;
    const cols = this.grid[0].length;
    const cp   = this.cellPx;
    const ox   = this.offsetX;
    const oy   = this.offsetY;

    // Validation border (red outline when unreachable)
    if (!this.valid) {
      ctx.strokeStyle = COL_INVALID;
      ctx.lineWidth = 3;
      ctx.strokeRect(ox - 2, oy - 2, cols * cp + 4, rows * cp + 4);
    }

    // Draw cells
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = ox + c * cp;
        const y = oy + r * cp;
        const isWall = this.grid[r][c] === 1;

        ctx.fillStyle = isWall ? COL_WALL : COL_PATH;
        ctx.fillRect(x, y, cp, cp);

        // Grid lines
        ctx.strokeStyle = COL_GRID_LINE;
        ctx.lineWidth = 1;
        ctx.strokeRect(x, y, cp, cp);

        // Border cells (non-editable) get a dim inset outline
        if (r === 0 || r === rows - 1 || c === 0 || c === cols - 1) {
          ctx.strokeStyle = COL_BORDER;
          ctx.lineWidth = 2;
          ctx.strokeRect(x + 1, y + 1, cp - 2, cp - 2);
        }
      }
    }

    // Markers: easy starts
    for (const [sr, sc] of this.starts) {
      this._drawMarker(sr, sc, COL_START, 'S');
    }

    // Marker: hard start
    if (this.hardStart) {
      this._drawMarker(this.hardStart[0], this.hardStart[1], COL_HARD, 'H');
    }

    // Marker: goal
    this._drawMarker(this.goal[0], this.goal[1], COL_GOAL, 'G');

    // Hover highlight
    if (this.hoverCell) {
      const [hr, hc] = this.hoverCell;
      ctx.fillStyle = COL_HOVER;
      ctx.fillRect(ox + hc * cp, oy + hr * cp, cp, cp);
    }

    // Instructions bar
    this._drawInstructions();

    // Validity warning text below grid
    if (!this.valid) {
      ctx.fillStyle = COL_INVALID;
      ctx.font = 'bold 14px Inter, Helvetica, sans-serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'top';
      ctx.fillText('Some starts cannot reach the goal!', cw / 2, oy + rows * cp + 10);
    }
  }

  /** Draw a coloured marker with a label letter inside a cell. */
  _drawMarker(r, c, colour, label) {
    const ctx = this.ctx;
    const cp  = this.cellPx;
    const x = this.offsetX + c * cp;
    const y = this.offsetY + r * cp;

    ctx.fillStyle = colour;
    ctx.globalAlpha = 0.35;
    ctx.fillRect(x, y, cp, cp);
    ctx.globalAlpha = 1.0;

    ctx.fillStyle = colour;
    ctx.font = `bold ${Math.round(cp * 0.5)}px Inter, Helvetica, sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, x + cp / 2, y + cp / 2);
  }

  /** Draw instruction text above the grid. */
  _drawInstructions() {
    const ctx = this.ctx;
    ctx.fillStyle = COL_TEXT_DIM;
    ctx.font = '12px Inter, Helvetica, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    ctx.fillText(
      'Click: wall/path  |  Shift+click: start  |  Alt+click: hard start  |  Ctrl/Cmd+click: goal  |  Press E or Escape to finish',
      this.canvas.width / 2,
      this.offsetY - 8,
    );
  }

  // ── Layout ────────────────────────────────────────────────────────────────

  _layout() {
    const rows = this.grid.length;
    const cols = this.grid[0].length;
    this.offsetX = Math.floor((this.canvas.width  - cols * this.cellPx) / 2);
    this.offsetY = Math.floor((this.canvas.height - rows * this.cellPx) / 2);
  }

  /** Convert canvas-space pixel to grid [row, col] or null. */
  _pixelToCell(px, py) {
    const r = Math.floor((py - this.offsetY) / this.cellPx);
    const c = Math.floor((px - this.offsetX) / this.cellPx);
    const rows = this.grid.length;
    const cols = this.grid[0].length;
    if (r >= 0 && r < rows && c >= 0 && c < cols) return [r, c];
    return null;
  }

  _isBorder(r, c) {
    return r === 0 || c === 0 ||
           r === this.grid.length - 1 ||
           c === this.grid[0].length - 1;
  }

  // ── Mouse handlers ────────────────────────────────────────────────────────

  /** Translate DOM event coordinates to canvas-space accounting for CSS scaling. */
  _canvasCoords(e) {
    const rect = this.canvas.getBoundingClientRect();
    const scaleX = this.canvas.width  / rect.width;
    const scaleY = this.canvas.height / rect.height;
    return [
      (e.clientX - rect.left) * scaleX,
      (e.clientY - rect.top)  * scaleY,
    ];
  }

  _handleMouseMove(e) {
    const [px, py] = this._canvasCoords(e);
    this.hoverCell = this._pixelToCell(px, py);

    // Drag-painting walls / paths
    if (this.painting && this.hoverCell) {
      const [r, c] = this.hoverCell;
      if (!this._isBorder(r, c) && !this._hasMarker(r, c)) {
        this.grid[r][c] = this.paintVal;
        this._validate();
      }
    }
    this.draw();
  }

  _handleMouseDown(e) {
    e.preventDefault();
    const [px, py] = this._canvasCoords(e);
    const cell = this._pixelToCell(px, py);
    if (!cell) return;

    const [r, c] = cell;
    if (this._isBorder(r, c)) return;

    if (e.shiftKey) {
      // Toggle easy start at this path cell
      this._toggleStart(r, c);
    } else if (e.altKey) {
      // Place hard start
      if (this.grid[r][c] === 0) {
        this.hardStart = [r, c];
      }
    } else if (e.ctrlKey || e.metaKey) {
      // Place goal
      if (this.grid[r][c] === 0) {
        this.goal = [r, c];
      }
    } else {
      // Wall / path toggle + begin drag-paint
      if (this._hasMarker(r, c)) return;
      const newVal = this.grid[r][c] === 1 ? 0 : 1;
      this.grid[r][c] = newVal;
      this.painting = true;
      this.paintVal = newVal;
    }

    this._validate();
    this.draw();
  }

  _handleMouseUp() {
    this.painting = false;
  }

  _handleMouseLeave() {
    this.hoverCell = null;
    this.painting  = false;
    this.draw();
  }

  // ── Marker helpers ────────────────────────────────────────────────────────

  /** Check whether (r, c) holds any special marker. */
  _hasMarker(r, c) {
    if (this.goal[0] === r && this.goal[1] === c) return true;
    if (this.hardStart && this.hardStart[0] === r && this.hardStart[1] === c) return true;
    for (const s of this.starts) {
      if (s[0] === r && s[1] === c) return true;
    }
    return false;
  }

  /** Add or remove an easy-start marker at (r, c). */
  _toggleStart(r, c) {
    if (this.grid[r][c] !== 0) return;
    const idx = this.starts.findIndex(s => s[0] === r && s[1] === c);
    if (idx !== -1) {
      if (this.starts.length > 1) {
        this.starts.splice(idx, 1);
      }
    } else {
      this.starts.push([r, c]);
    }
  }

  // ── Validation ────────────────────────────────────────────────────────────

  /** Check BFS reachability from every start (and hard start) to the goal. */
  _validate() {
    this.valid = true;
    const allStarts = [...this.starts];
    if (this.hardStart) allStarts.push(this.hardStart);

    for (const s of allStarts) {
      if (!bfsShortestPath(this.grid, s, this.goal)) {
        this.valid = false;
        return;
      }
    }
  }
}

// ── Utility ─────────────────────────────────────────────────────────────────

function deepCopyGrid(grid) {
  return grid.map(row => [...row]);
}
