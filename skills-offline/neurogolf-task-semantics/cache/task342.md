# task342 Semantics

## Sources

- Current champion builder: `solutions_py/task342.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task342.json`
- ARC-GEN task id: `d89b689b`
- ARC-DSL task id: `d89b689b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d89b689b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d89b689b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task342.py`

## Pattern

The grid is always `10x10`. The input contains a cyan (`8`) `2x2` square whose top-left corner is `(brow,bcol)`, with `brow,bcol` in `2..7`. Four non-cyan singleton marker pixels are placed in the four diagonal regions around the square:

- marker 0: above and left of the square;
- marker 1: above and right of the square;
- marker 2: below and left of the square;
- marker 3: below and right of the square.

The output removes the singleton markers and replaces the cyan square with those four marker colors in matching quadrant order:

```text
out[brow    ][bcol    ] = top-left marker color
out[brow    ][bcol + 1] = top-right marker color
out[brow + 1][bcol    ] = bottom-left marker color
out[brow + 1][bcol + 1] = bottom-right marker color
```

All other cells are background `0`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    cyan = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 8]
    brow = min(r for r, _ in cyan)
    bcol = min(c for _, c in cyan)
    out = [[0 for _ in row] for row in grid]
    markers = []
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v not in (0, 8):
                markers.append((r, c, v))
    for r, c, v in markers:
        rr = 0 if r < brow else 1
        cc = 0 if c < bcol else 1
        out[brow + rr][bcol + cc] = v
    return out
```

## Generator Constraints

- Grid size is fixed at `10x10`.
- The cyan square top-left corner satisfies `2 <= brow <= 7` and `2 <= bcol <= 7`, so every quadrant region has at least two rows/columns of space.
- Exactly four singleton marker pixels are generated, one in each diagonal region around the cyan square.
- Marker colors are four random non-cyan colors, sampled with `common.random_colors(4, exclude=[8])`; they may include any non-cyan digit and are assigned to the four quadrants in generator order.
- The output contains only the recolored 2x2 square; all original marker locations become background.

## Reference Notes

ARC-DSL selects the cyan set, filters singleton objects, assigns each singleton to the nearest cyan cell by Manhattan distance, recolors that nearest cyan cell with the singleton color, covers the original singletons, and paints the recolored square. The Code Golf solution rotates the grid repeatedly and uses a regex-style transfer to move each marker color into the matching cyan cell; this is an obfuscated version of the same quadrant/nearest-cell rule.

There is no ambiguity in the references: the four markers are guaranteed to correspond to the four cyan square cells by quadrant.
