# task303 Semantics

## Sources

- Current champion builder: `solutions_py/task303.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task303.json`
- ARC-GEN task id: `c1d99e64`
- ARC-DSL task id: `c1d99e64`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c1d99e64.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c1d99e64.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task303.py`

## Pattern

The input is a rectangular grid, width and height each between `10` and `30`. It contains black (`0`) and one non-red foreground color. Some entire rows and some entire columns have been overwritten to black. The output is the same grid except every cell lying on one of those all-black rows or all-black columns is recolored red (`2`). Other cells keep their original black/foreground value.

Equivalently, find every row whose cells are all black and every column whose cells are all black, take the union of those row/column coordinates, and fill that union with red.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    full_rows = [all(v == 0 for v in row) for row in grid]
    full_cols = [all(grid[r][c] == 0 for r in range(h)) for c in range(w)]
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if full_rows[r] or full_cols[c]:
                out[r][c] = 2
    return out
```

## Generator Constraints

- Input width and height are independently sampled in `[10, 30]`.
- The nonblack foreground color is a single random color from `1..9` excluding red (`2`).
- Before the special lines are drawn, each row and each column is forced to contain at least one black cell and at least one foreground-colored cell.
- Then `0..3` internal rows and `0..3` internal columns are selected from coordinates `1..height-2` and `1..width-2` and set entirely to black.
- Because the pre-line grid has every row/column mixed, any all-black row or column in the final input is guaranteed to be one of the selected line coordinates.
- There may be no selected rows or no selected columns, but the reference examples include both row-only, column-only, and row+column cases.

## Reference Notes

The ARC-DSL solver calls `frontiers(I)` to identify complete horizontal and vertical black frontiers, merges their cells, and fills those cells with color `TWO`. The Code Golf expression performs the same test compactly by keeping an input cell only when both its row and column are not all-zero; otherwise it emits `2`.
