# task047 Semantics

## Sources

- Current champion builder: `solutions_py/task047.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task047.json`
- ARC-GEN task id: `23581191`
- ARC-DSL task id: `23581191`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_23581191.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_23581191.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task047.py`

## Pattern

The input is a square grid, normally 9x9, containing exactly two colored single
pixels: one cyan (`8`) and one orange (`7`). The output draws the full horizontal
and vertical frontier through each pixel in that pixel's color. The two
off-diagonal intersections between the cyan row/column and orange row/column are
colored red (`2`). The original two pixel centers keep their own colors.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    points = []
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color:
                points.append((r, c, color))
    out = [row[:] for row in grid]
    for r, c, color in points:
        for cc in range(w):
            out[r][cc] = color
        for rr in range(h):
            out[rr][c] = color
    if len(points) == 2:
        r0, c0, _ = points[0]
        r1, c1, _ = points[1]
        out[r0][c1] = 2
        out[r1][c0] = 2
    return out
```

## Generator Constraints

ARC-GEN defaults to a 9x9 square. It samples two distinct rows and two distinct
columns from `1..size-3`, so the points are never on the outer border and never
on row/column `size-2` or `size-1`. The first sampled point is cyan and the
second is orange. The two row coordinates are distinct and the two column
coordinates are distinct because `common.sample(..., 2)` is used.

## Reference Notes

The ARC-DSL solver finds the two foreground objects, builds the vertical and horizontal frontiers through each center, recolors those frontiers by the object's color, paints them on the input, then fills the intersection of the two frontier sets with red. The Code Golf solution uses row and column sums modulo 13 to exploit the fixed colors `8`, `7`, and red `2`; it confirms that there are only two singleton markers and no object-shape ambiguity.
