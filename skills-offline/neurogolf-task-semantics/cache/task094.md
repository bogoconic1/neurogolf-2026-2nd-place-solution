# task094 Semantics

## Sources

- Current champion builder: `solutions_py/task094.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task094.json`
- ARC-GEN task id: `41e4d17e`
- ARC-DSL task id: `41e4d17e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_41e4d17e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_41e4d17e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task094.py`

## Pattern

The input is a fixed 15 by 15 cyan (`8`) grid containing zero, one, or two hollow blue (`1`) 5 by 5 square boxes. Each box is centered at a row and column in `3..11`. The output keeps the blue box borders fixed and draws magenta (`6`) crosshair lines through each box center: the entire center row and entire center column become magenta, except original blue border cells remain blue because the box borders are repainted over the crosshairs. Cyan remains cyan everywhere else.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    blue = 1
    pink = 6
    out = [row[:] for row in grid]
    seen = [[False] * w for _ in range(h)]
    centers = []
    blue_cells = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] != blue:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == blue:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            blue_cells.extend(comp)
            rows = [rr for rr, _ in comp]
            cols = [cc for _, cc in comp]
            centers.append(((min(rows) + max(rows)) // 2, (min(cols) + max(cols)) // 2))

    for r, c in centers:
        for j in range(w):
            out[r][j] = pink
        for i in range(h):
            out[i][c] = pink
    for r, c in blue_cells:
        out[r][c] = blue
    return out
```

## Generator Constraints

ARC-GEN uses a fixed square size `15`. Candidate centers are sampled from rows and columns `3..11`, so every 5 by 5 box fits with at least one cell of margin. It attempts to place two boxes, but skips a sampled second box if it overlaps a previous box or if the crosshair rows/columns would be too close: centers with row distance `< 3`, column distance `< 3`, or both row and column distance `< 6` are rejected. As a result generated examples can contain one or two boxes, and possibly zero if both random attempts were skipped, though the bundled examples contain one or two. Colors are fixed: background cyan `8`, box border blue `1`, and crosshair fill pink/magenta `6`.

## Reference Notes

The ARC-DSL solver finds non-background 4-connected objects, maps each object to the union of its vertical and horizontal frontiers through its center (`vfrontier` and `hfrontier` composed with `center`), merges those crosshair cells, and underfills them with color `6`. `underfill` preserves existing non-background blue border cells, matching the generator order where crosshairs are drawn first and blue borders are written afterward. The Code Golf solution is a compact recursive/string approach that detects the blue box pattern and substitutes `6` along the matching row/column structure; it agrees with the same crosshair-through-box-center rule.
