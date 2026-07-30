# task248 Semantics

## Sources

- Current champion builder: `solutions_py/task248.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task248.json`
- ARC-GEN task id: `a3df8b1e`
- ARC-DSL task id: `a3df8b1e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a3df8b1e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a3df8b1e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task248.py`

## Pattern

The grid is black with a single blue (`1`) seed at the bottom-left. The output draws a one-cell-wide blue diagonal ray that bounces between the left and right vertical walls as it travels upward/down the displayed rows. For row `r` from top to bottom in a height-10 grid of width `w`, the blue column is a triangular-wave position with period `2*w - 2`, matching repeated reflection between columns `0` and `w-1`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    period = 2 * w - 2
    out = [[0 for _ in range(w)] for _ in range(h)]
    i = h
    for r in range(h):
        i = (i - 1) % period
        c = min(i, period - i)
        out[r][c] = 1
    return out
```

## Generator Constraints

ARC-GEN fixes height to `10` and samples width in `[2, 10]`. Colors are black background (`0`) and blue path (`1`). The input contains only the initial blue seed from the bounce construction; the output is the full reflected/bouncing path over the same `10 x width` shape. The submitted NeuroGolf tensor still has the standard `[1,10,30,30]` shape with no-cell padding outside the actual grid.

## Reference Notes

The ARC-DSL solver builds the ray from the seed, finds the reflected/bounce object, crops one row, vertically repeats it, mirrors/crops back to the original shape, and mirrors again. The Code Golf solution directly implements the triangular-wave column formula `c = min(i, period - i)` while decrementing `i` modulo `period` for each row.
