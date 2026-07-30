# task356 Semantics

## Sources

- Current champion builder: `solutions_py/task356.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task356.json`
- ARC-GEN task id: `ded97339`
- ARC-DSL task id: `ded97339`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ded97339.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ded97339.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task356.py`

## Pattern

The input is a `10 x 10` black grid with sparse cyan (`8`) points. The output keeps all input points and fills cyan along straight horizontal or vertical segments between every pair of cyan points that share the same row or the same column. Diagonal pairs do not connect. In effect, each row gets cyan filled from its leftmost cyan point to its rightmost cyan point when the row has at least two points, and each column gets cyan filled from its topmost cyan point to its bottommost cyan point when the column has at least two points.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        cols = [c for c in range(w) if grid[r][c] == 8]
        if cols:
            for c in range(min(cols), max(cols) + 1):
                out[r][c] = 8
    for c in range(w):
        rows = [r for r in range(h) if grid[r][c] == 8]
        if rows:
            for r in range(min(rows), max(rows) + 1):
                out[r][c] = 8
    return out
```

## Generator Constraints

- Grid size is always `10 x 10`.
- Input has at least one cyan point.
- Points are sampled sparsely with probability about 0.05.
- Only colors are black (`0`) and cyan (`8`).
- Horizontal/vertical filling is inclusive of endpoints; rows or columns with a single point are unchanged except for that point.

## Reference Notes

The ARC-DSL solver forms all pairs of cyan points, connects each pair, filters those connections to vertical or horizontal lines, and underfills cyan on their union. The Code Golf solution computes row and column prefix/suffix evidence and ORs the two directions, matching the same span-fill rule.
