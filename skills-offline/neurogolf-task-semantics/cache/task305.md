# task305 Semantics

## Sources

- Current champion builder: `solutions_py/task305.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task305.json`
- ARC-GEN task id: `c3f564a4`
- ARC-DSL task id: `c3f564a4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c3f564a4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c3f564a4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task305.py`

## Pattern

The true output is a 16x16 diagonal periodic stripe grid. For a period `P` in `4..9`, each cell has color `(row + col) % P + 1`. The input starts from that full stripe grid and then blacks out several rectangular cutouts. The task is to restore the missing stripe colors.

The generator guarantees that after cutouts, every nonzero stripe color is still visible in every row and every column. This makes the period and row/column color relation recoverable from visible cells.

## Readable Python Solver

```python
def solve(grid):
    colors = sorted({v for row in grid for v in row if v})
    period = len(colors)
    h, w = len(grid), len(grid[0])
    return [[(r + c) % period + 1 for c in range(w)] for r in range(h)]
```

## Generator Constraints

- Grid size is fixed at `16x16`.
- Period/color count is randomly chosen from `4..9`.
- Base grid is `(row + col) % colors + 1`.
- The input is corrupted by several black rectangles. In random generation there are five cutouts, each width and height `2..4`; hand examples include four or five cutouts.
- The generator rejects cases where any row or column loses visibility of a stripe color, so every row and every column still contains all colors `1..P` somewhere outside cutouts.
- Color 0 is cutout/background only and is not part of the restored stripe palette.

## Reference Notes

The ARC-DSL solution mirrors the grid, takes per-cell maxima to pair visible diagonal stripe evidence, identifies black cutouts, and paints shifted copies of the visible stripe object to cover them. The Code Golf solution extracts the nonzero palette from the first row and tiles it diagonally. Both match the simple period-based stripe restoration rule.
