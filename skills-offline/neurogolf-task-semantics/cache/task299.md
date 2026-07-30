# task299 Semantics

## Sources

- Current champion builder: `solutions_py/task299.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task299.json`
- ARC-GEN task id: `bdad9b1f`
- ARC-DSL task id: `bdad9b1f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_bdad9b1f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_bdad9b1f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task299.py`

## Pattern

The input is a 6x6 grid containing two partial street hints: a red horizontal line segment in one row across columns 0 and 1, and a cyan vertical line segment in one column across rows 0 and 1. The target completes the full red row and full cyan column across the whole grid, and colors their intersection yellow.

The generator may horizontally flip the whole grid, so the red clue can appear on the left or right edge. The cyan clue remains in the target column after the same flip. The row and column are each chosen from indices 2..4.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    red = 2
    yellow = 4
    cyan = 8

    row = next(r for r in range(h) if any(grid[r][c] == red for c in range(w)))
    col = next(c for c in range(w) if any(grid[r][c] == cyan for r in range(h)))

    out = [list(r) for r in grid]
    for c in range(w):
        out[row][c] = red
    for r in range(h):
        out[r][col] = cyan
    out[row][col] = yellow
    return out
```

## Generator Constraints

- Grid size is fixed at 6x6.
- Red is color `2`, yellow is color `4`, cyan is color `8`.
- The red row index and cyan column index are each sampled from `2..4`.
- In the unflipped input, red appears only at `(row, 0)` and `(row, 1)`, while cyan appears only at `(0, col)` and `(1, col)`.
- With `flip=True`, both input and output are horizontally flipped. This moves the red clue to the right edge and maps the cyan column to `5 - col`.
- The output always contains exactly one full red row, one full cyan column, and a yellow intersection.

## Reference Notes

The ARC-DSL solver finds the center of the red object and cyan object, builds a horizontal frontier through the red center and a vertical frontier through the cyan center, fills those with red and cyan, and then fills their intersection with yellow. The Code Golf solution encodes the same operation as a compact row/column rule over color membership.
