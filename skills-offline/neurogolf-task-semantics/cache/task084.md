# task084 Semantics

## Sources

- Current champion builder: `solutions_py/task084.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task084.json`
- ARC-GEN task id: `3bd67248`
- ARC-DSL task id: `3bd67248`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3bd67248.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3bd67248.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task084.py`

## Pattern

The input is a square grid whose first column is filled with one non-background color and whose remaining cells are black. The output preserves the first column, paints the bottom row from column 1 through the right edge yellow (`4`), and paints the anti-diagonal from `(n-2, 1)` up to `(0, n-1)` red (`2`). The bottom-left cell remains the original first-column color, so yellow starts at column 1.

There is no tie-breaking. The geometry depends only on the square size `n`.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    out = [row[:] for row in grid]
    for c in range(1, n):
        out[n - 1][c] = 4
        out[n - 1 - c][c] = 2
    return out
```

## Generator Constraints

ARC-GEN samples square size `n` from 3..21 and a left-column color from 1..9. It creates an `n x n` grid and output. The input has that color in every row of column 0 and black elsewhere. The output uses the same left column, yellow along the bottom row except column 0, and red along the anti-diagonal one row above the bottom-left through the top-right corner.

The NeuroGolf tensor still has standard `[1,10,30,30]` shape; true grids are square and occupy the top-left `n x n` region.

## Reference Notes

ARC-DSL derives the height, shoots an up-right ray from `(n-3,1)`/right ray from `(n-2,1)` through its DSL coordinates, then fills red/yellow. The Code Golf solution destructures the final row and all preceding rows, then walks from the right edge leftward setting red on preceding rows and yellow on the last row. Both references confirm the same anti-diagonal plus bottom-row rule.
