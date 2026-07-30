# task350 Semantics

## Sources

- Current champion builder: `solutions_py/task350.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task350.json`
- ARC-GEN task id: `dbc1a6ce`
- ARC-DSL task id: `dbc1a6ce`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_dbc1a6ce.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_dbc1a6ce.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task350.py`

## Pattern

The input contains blue (`1`) pixels on a black background. The output keeps all
blue pixels and fills cyan (`8`) on background cells that lie on a horizontal or
vertical segment between two blue pixels in the same row or same column. Blue
endpoints stay blue; only background cells under the connecting row/column
segments become cyan.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    blues = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 1]

    for r1, c1 in blues:
        for r2, c2 in blues:
            if r1 == r2:
                for c in range(min(c1, c2), max(c1, c2) + 1):
                    if out[r1][c] == 0:
                        out[r1][c] = 8
            if c1 == c2:
                for r in range(min(r1, r2), max(r1, r2) + 1):
                    if out[r][c1] == 0:
                        out[r][c1] = 8
    return out
```

## Generator Constraints

ARC-GEN chooses width `8..24` and height `width + {-2,-1,0,1,2}`, then places
random blue pixels with about 10% density. The fixed validation examples include
multiple blue pixels in the same row and column, edge rows/columns, rectangular
and nearly square grids, and larger grids up to `21x19`. The NeuroGolf tensor is
still padded to `[1,10,30,30]`; meaningful content is inside the generated grid.

## Reference Notes

The ARC-DSL solver takes all pairs of blue cells, connects each pair, filters to horizontal or vertical lines, and underfills those line cells with cyan (`8`) so blue endpoints are preserved. The Code Golf 2025 solution performs the same operation by scanning rows and then columns and turning zeros between repeated blue markers into cyan.
