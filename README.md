# MaxRL vs REINFORCE -- Live Training Demo

Interactive visualization comparing standard REINFORCE (expected reward) against Maximum Likelihood RL from [Tajwar et al. 2025](https://arxiv.org/abs/2602.02710).

**Live demo**: [lcrh.github.io/maxrl_demo](https://lcrh.github.io/maxrl_demo)

## What it shows

- **RL** maximizes E_maze[ p(success|maze) ] -- hard mazes contribute little gradient
- **MaxRL** maximizes E_maze[ log p(success|maze) ] -- hard mazes matter equally

A tabular softmax policy navigates grid mazes. With a mix of easy and hard starting positions, RL only learns the easy start while MaxRL learns both.

## Features

- Side-by-side live training with heatmaps and path traces
- Single-start and multi-start modes
- Default grid, random maze generation, or draw your own
- Adjustable speed and easy/hard start ratio

## Run locally

Open `index.html` directly, or serve with:

```
npx serve .
```

No build step, no dependencies. Pure ES modules + Canvas API.
