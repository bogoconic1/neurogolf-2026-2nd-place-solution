# task050 Semantics

## Sources

- Current champion builder: `solutions_py/task050.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task050.json`
- ARC-GEN task id: `253bf280`
- ARC-DSL task id: `253bf280`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_253bf280.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_253bf280.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task050.py`

## Pattern

The input is a small grid padded into the standard NeuroGolf tensor. It contains cyan (`8`) points. Any two cyan points that share a row or share a column define a straight segment. The output keeps every original cyan point cyan and fills the cells strictly between each aligned pair with green (`3`). Points that do not share a row or column with another point remain isolated cyan points. The output size is the same as the input grid, padded in NeuroGolf to `[1,10,30,30]`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 8]
    for i, (r1, c1) in enumerate(pts):
        for r2, c2 in pts[:i]:
            if r1 == r2:
                a, b = sorted((c1, c2))
                for c in range(a + 1, b):
                    out[r1][c] = 3
            if c1 == c2:
                a, b = sorted((r1, r2))
                for r in range(a + 1, b):
                    out[r][c1] = 3
    for r, c in pts:
        out[r][c] = 8
    return out
```

## Generator Constraints

ARC-GEN samples input width and height from `3..15`. It attempts up to three non-crossing horizontal or vertical cyan line endpoint pairs, then up to two isolated cyan points. For a horizontal pair, endpoints share one row and have non-adjacent columns; the generator rejects crossings, duplicate endpoint rows, and duplicate endpoint columns. For a vertical pair, endpoints share one column and have non-adjacent rows; it rejects crossings, duplicate endpoint columns, and duplicate endpoint rows. Isolated points are not placed on any existing endpoint row or endpoint column. Hand examples include one-point, one horizontal segment, one vertical segment, two disjoint segments, and mixtures with isolated points.

## Reference Notes

The ARC-DSL solver gets all cyan cells, connects every pair, filters connected sets whose size is greater than one and that are horizontal or vertical lines, fills those cells green, then restores the original cyan cells. The Code Golf solution is a compact row/column scan that propagates green between aligned cyan endpoints. Both agree that endpoints remain cyan and only same-row or same-column gaps are filled green.
