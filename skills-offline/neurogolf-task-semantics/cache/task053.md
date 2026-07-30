# task053 Semantics

## Sources

- Current champion builder: `solutions_py/task053.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task053.json`
- ARC-GEN task id: `25ff71a9`
- ARC-DSL task id: `25ff71a9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_25ff71a9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_25ff71a9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task053.py`

## Pattern

The input is a `3x3` grid with one small same-color object made of one to three cells in the top two rows. The output moves the colored cells down by one row. The top row becomes black/background, old row 0 becomes output row 1, and old row 1 becomes output row 2. The object color is preserved.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h - 1):
        for c in range(w):
            out[r + 1][c] = grid[r][c]
    return out
```

## Generator Constraints

ARC-GEN uses size `3`. It samples one to three colored cells from rows `0..1` only and color `1` or `2`. The bottom row of the input is always background. Output places each sampled cell one row lower, so no colored cell moves out of bounds. Hand examples include full top/middle rows, L-shapes, and single-cell cases.

## Reference Notes

The ARC-DSL solver identifies the object and moves it `DOWN`. The Code Golf solution duplicates the row list and returns `(2*g)[2:5]`, which is equivalent to `[old row 2, old row 0, old row 1]`; because old row 2 is empty, this is a one-row downward shift.
