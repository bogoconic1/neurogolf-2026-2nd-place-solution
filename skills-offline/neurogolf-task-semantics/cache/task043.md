# task043 Semantics

## Sources

- Current champion builder: `solutions_py/task043.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task043.json`
- ARC-GEN task id: `2281f1f4`
- ARC-DSL task id: `2281f1f4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_2281f1f4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_2281f1f4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task043.py`

## Pattern

The visible task grid is 10x10. The input is black except for gray (`5`) markers on the top row and gray markers on the right edge. Top-row markers select columns. Right-edge markers select rows. The output keeps all gray markers, paints red (`2`) at every selected row/selected column intersection, keeps the remaining visible cells black (`0`), and leaves the fixed NeuroGolf padding outside the 10x10 working area zero-hot.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    cols = [c for c in range(w - 1) if grid[0][c] == 5]
    rows = [r for r in range(1, h) if grid[r][w - 1] == 5]
    out = [row[:] for row in grid]
    for r in rows:
        for c in cols:
            out[r][c] = 2
    return out
```

## Generator Constraints

ARC-GEN uses square 10x10 grids. It samples a subset of rows from `1..9` and a subset of columns from `0..8`; either subset may be empty. It writes gray markers at `(0, c)` for selected columns and `(r, 9)` for selected rows. The output starts as the input, then writes red at every Cartesian-product intersection `(r, c)` for selected rows and columns. There are no other foreground colors or objects.

## Reference Notes

The ARC-DSL solution takes all gray cells, forms their Cartesian product, maps each pair to the tuple of the first coordinate from one and the second coordinate from the other, removes the upper-right marker corner, and underfills the resulting cells with red. The Code Golf solution is a compact row/column marker product expression; it confirms that the rule is a marker-grid Cartesian product, not an object-shape transformation.
