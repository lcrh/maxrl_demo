// engine.js – ES module port of environment.py + training.py

// Actions
export const UP = 0;
export const DOWN = 1;
export const LEFT = 2;
export const RIGHT = 3;
export const ACTION_DELTAS = [
  [-1, 0], // UP
  [1, 0],  // DOWN
  [0, -1], // LEFT
  [0, 1],  // RIGHT
];

export const MAX_LOGIT = 20.0;

// 0=path, 1=wall
// Three Paths maze (11x11)
export const MAZE_TRAIN = [
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
  [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
  [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
  [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
  [1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1],
  [1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1],
  [1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

// Test maze: left corridor blocked
export const MAZE_TEST = [
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
  [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
  [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
  [1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
  [1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1],
  [1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1],
  [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
];

// ---------------------------------------------------------------------------
// GridWorld
// ---------------------------------------------------------------------------

export class GridWorld {
  constructor(grid, start = [1, 1], goal = [9, 9]) {
    this.grid = grid;
    this.height = grid.length;
    this.width = grid[0].length;
    this.start = start;
    this.goal = goal;
  }

  inBounds(row, col) {
    return row >= 0 && row < this.height && col >= 0 && col < this.width;
  }

  isPath(row, col) {
    return this.inBounds(row, col) && this.grid[row][col] === 0;
  }

  step(pos, action) {
    const [row, col] = pos;
    const [dr, dc] = ACTION_DELTAS[action];
    const nr = row + dr;
    const nc = col + dc;
    let newPos;
    if (this.isPath(nr, nc)) {
      newPos = [nr, nc];
    } else {
      newPos = [row, col];
    }
    const done = newPos[0] === this.goal[0] && newPos[1] === this.goal[1];
    return { newPos, done };
  }
}

// ---------------------------------------------------------------------------
// TabularSoftmaxPolicy
// ---------------------------------------------------------------------------

export class TabularSoftmaxPolicy {
  constructor(height, width, nActions = 4) {
    this.height = height;
    this.width = width;
    this.nActions = nActions;
    this.logits = new Float64Array(height * width * nActions);
  }

  _idx(row, col, action) {
    return (row * this.width + col) * this.nActions + action;
  }

  getProbs(row, col) {
    const base = (row * this.width + col) * this.nActions;
    // Find max for numerical stability
    let maxZ = -Infinity;
    for (let a = 0; a < this.nActions; a++) {
      const v = this.logits[base + a];
      if (v > maxZ) maxZ = v;
    }
    const probs = new Float64Array(this.nActions);
    let sum = 0;
    for (let a = 0; a < this.nActions; a++) {
      const e = Math.exp(this.logits[base + a] - maxZ);
      probs[a] = e;
      sum += e;
    }
    for (let a = 0; a < this.nActions; a++) {
      probs[a] /= sum;
    }
    return probs;
  }

  sampleAction(row, col) {
    const probs = this.getProbs(row, col);
    const r = Math.random();
    let cumulative = 0;
    for (let a = 0; a < this.nActions; a++) {
      cumulative += probs[a];
      if (r < cumulative) return a;
    }
    return this.nActions - 1;
  }

  scoreFunction(row, col, action) {
    const probs = this.getProbs(row, col);
    const grad = new Float64Array(this.nActions);
    for (let a = 0; a < this.nActions; a++) {
      grad[a] = -probs[a];
    }
    grad[action] += 1.0;
    return grad;
  }

  clipLogits() {
    for (let r = 0; r < this.height; r++) {
      for (let c = 0; c < this.width; c++) {
        const base = (r * this.width + c) * this.nActions;
        let maxVal = -Infinity;
        for (let a = 0; a < this.nActions; a++) {
          const v = this.logits[base + a];
          if (v > maxVal) maxVal = v;
        }
        for (let a = 0; a < this.nActions; a++) {
          let v = this.logits[base + a] - maxVal;
          if (v < -MAX_LOGIT) v = -MAX_LOGIT;
          if (v > MAX_LOGIT) v = MAX_LOGIT;
          this.logits[base + a] = v;
        }
      }
    }
  }

  copy() {
    const p = new TabularSoftmaxPolicy(this.height, this.width, this.nActions);
    p.logits.set(this.logits);
    return p;
  }
}

// ---------------------------------------------------------------------------
// Rollout
// ---------------------------------------------------------------------------

export function rollout(policy, env, start, maxSteps = 80) {
  let pos = start || env.start;
  const stateActions = [];
  const path = [[pos[0], pos[1]]];

  for (let t = 0; t < maxSteps; t++) {
    if (pos[0] === env.goal[0] && pos[1] === env.goal[1]) {
      return { stateActions, reachedGoal: true, path };
    }
    const action = policy.sampleAction(pos[0], pos[1]);
    stateActions.push([pos[0], pos[1], action]);
    const result = env.step(pos, action);
    pos = result.newPos;
    path.push([pos[0], pos[1]]);
  }

  const reachedGoal = pos[0] === env.goal[0] && pos[1] === env.goal[1];
  return { stateActions, reachedGoal, path };
}

// ---------------------------------------------------------------------------
// Training updates
// ---------------------------------------------------------------------------

function _chooseWeighted(probs) {
  const r = Math.random();
  let cumulative = 0;
  for (let i = 0; i < probs.length; i++) {
    cumulative += probs[i];
    if (r < cumulative) return i;
  }
  return probs.length - 1;
}

export function reinforceUpdate(policy, env, starts, startProbs, N = 16, lr = 0.5, maxSteps = 80) {
  const trajectories = [];
  for (let i = 0; i < N; i++) {
    const idx = _chooseWeighted(startProbs);
    const traj = rollout(policy, env, starts[idx], maxSteps);
    trajectories.push(traj);
  }

  const rewards = new Float64Array(N);
  let K = 0;
  for (let i = 0; i < N; i++) {
    rewards[i] = trajectories[i].reachedGoal ? 1.0 : 0.0;
    K += rewards[i];
  }
  const rHat = K / N;

  const grad = new Float64Array(policy.logits.length);

  for (let i = 0; i < N; i++) {
    const advantage = rewards[i] - rHat;
    const sa = trajectories[i].stateActions;
    for (let j = 0; j < sa.length; j++) {
      const [row, col, action] = sa[j];
      const score = policy.scoreFunction(row, col, action);
      const base = (row * policy.width + col) * policy.nActions;
      for (let a = 0; a < policy.nActions; a++) {
        grad[base + a] += (1.0 / N) * advantage * score[a];
      }
    }
  }

  for (let i = 0; i < policy.logits.length; i++) {
    policy.logits[i] += lr * grad[i];
  }
  policy.clipLogits();
  return K;
}

export function maxrlUpdate(policy, env, starts, startProbs, N = 16, lr = 0.5, maxSteps = 80) {
  const trajectories = [];
  for (let i = 0; i < N; i++) {
    const idx = _chooseWeighted(startProbs);
    const traj = rollout(policy, env, starts[idx], maxSteps);
    trajectories.push(traj);
  }

  const rewards = new Float64Array(N);
  let K = 0;
  for (let i = 0; i < N; i++) {
    rewards[i] = trajectories[i].reachedGoal ? 1.0 : 0.0;
    K += rewards[i];
  }

  if (K > 0) {
    const grad = new Float64Array(policy.logits.length);

    for (let i = 0; i < N; i++) {
      const weight = rewards[i] / K - 1.0 / N;
      const sa = trajectories[i].stateActions;
      for (let j = 0; j < sa.length; j++) {
        const [row, col, action] = sa[j];
        const score = policy.scoreFunction(row, col, action);
        const base = (row * policy.width + col) * policy.nActions;
        for (let a = 0; a < policy.nActions; a++) {
          grad[base + a] += weight * score[a];
        }
      }
    }

    for (let i = 0; i < policy.logits.length; i++) {
      policy.logits[i] += lr * grad[i];
    }
    policy.clipLogits();
  }

  return K;
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

export function evaluateFromStart(policy, env, start, nEval = 100, maxSteps = 25) {
  let successes = 0;
  for (let i = 0; i < nEval; i++) {
    const traj = rollout(policy, env, start, maxSteps);
    if (traj.reachedGoal) successes++;
  }
  return successes / nEval;
}

// ---------------------------------------------------------------------------
// BFS shortest path
// ---------------------------------------------------------------------------

export function bfsShortestPath(grid, start, goal) {
  const height = grid.length;
  const width = grid[0].length;
  const visited = Array.from({ length: height }, () => new Uint8Array(width));
  const queue = [[start[0], start[1], 0]];
  visited[start[0]][start[1]] = 1;

  // Parent tracking for path reconstruction: store [parentRow, parentCol] per cell
  const parent = Array.from({ length: height }, () =>
    Array.from({ length: width }, () => [-1, -1])
  );

  let head = 0;
  while (head < queue.length) {
    const [r, c, dist] = queue[head++];
    if (r === goal[0] && c === goal[1]) {
      const path = [];
      let cr = r, cc = c;
      while (cr !== -1) {
        path.push([cr, cc]);
        const [pr, pc] = parent[cr][cc];
        cr = pr;
        cc = pc;
      }
      path.reverse();
      return { distance: dist, path };
    }
    for (let a = 0; a < 4; a++) {
      const [dr, dc] = ACTION_DELTAS[a];
      const nr = r + dr;
      const nc = c + dc;
      if (nr >= 0 && nr < height && nc >= 0 && nc < width &&
          grid[nr][nc] === 0 && !visited[nr][nc]) {
        visited[nr][nc] = 1;
        parent[nr][nc] = [r, c];
        queue.push([nr, nc, dist + 1]);
      }
    }
  }
  return null; // No path found
}

// ---------------------------------------------------------------------------
// Random maze generator (recursive backtracker)
// ---------------------------------------------------------------------------

export function generateMaze(rows, cols, start = [1, 1], goal = null) {
  if (!goal) goal = [rows - 2, cols - 2];

  // Random scatter: border is walls, interior is random ~30% walls
  const wallProb = 0.30;
  const grid = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => {
      if (r === 0 || r === rows - 1 || c === 0 || c === cols - 1) return 1;
      return Math.random() < wallProb ? 1 : 0;
    })
  );

  // Ensure start and goal are open
  grid[start[0]][start[1]] = 0;
  grid[goal[0]][goal[1]] = 0;

  // If not connected, carve a random path via BFS from start toward goal
  // removing walls along the way until connected
  const check = bfsShortestPath(grid, start, goal);
  if (!check) {
    // BFS from start, tracking parents, until we can't expand
    // Then remove random walls adjacent to the frontier and retry
    const dirs = [[-1,0],[1,0],[0,-1],[0,1]];
    let connected = false;
    for (let attempt = 0; attempt < 200 && !connected; attempt++) {
      // Pick a random wall in the interior and clear it
      const r = 1 + Math.floor(Math.random() * (rows - 2));
      const c = 1 + Math.floor(Math.random() * (cols - 2));
      grid[r][c] = 0;
      if (bfsShortestPath(grid, start, goal)) connected = true;
    }
    // If still not connected after 200 pokes, just regenerate
    if (!connected) return generateMaze(rows, cols, start, goal);
  }

  return grid;
}
