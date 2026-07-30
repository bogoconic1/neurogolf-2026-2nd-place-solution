# task362 Semantics

## Sources

- Current champion builder: `solutions_py/task362.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task362.json`
- ARC-GEN task id: `e48d4e1a`
- ARC-DSL task id: `e48d4e1a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e48d4e1a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e48d4e1a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task362.py`

## Pattern

The input is a 10x10 black grid containing one non-gray colored cross: a full horizontal line at row `row` and a full vertical line at column `col`, both in the same color. The rightmost column also has `offset` gray cells at the top, where `offset` is 1, 2, or 3 after clipping to keep the shifted cross in bounds.

The output is a blank black 10x10 grid with the same colored cross shifted down and left by that offset: the horizontal line moves from `row` to `row + offset`, and the vertical line moves from `col` to `col - offset`. The gray offset markers and original cross are not preserved.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    offset = sum(1 for r in range(h) if grid[r][w - 1] == 5)
    color_counts = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v not in (0, 5):
                color_counts[v] = color_counts.get(v, 0) + 1
    color = max(color_counts, key=color_counts.get)

    rows = [r for r in range(h) if sum(grid[r][c] == color for c in range(w)) >= 4]
    cols = [c for c in range(w) if sum(grid[r][c] == color for r in range(h)) >= 4]
    row = rows[0]
    col = cols[0]

    out = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        out[row + offset][c] = color
    for r in range(h):
        out[r][col - offset] = color
    return out
```

## Generator Constraints

- Grid size is always 10x10.
- The cross color is chosen from colors 1..9 excluding gray 5.
- The original row and column are sampled from 1..8, so the input cross is not initially on an outer border.
- `offset` starts in 1..3 and is clipped by `size - row - 1`, `row`, and `col`; therefore the shifted horizontal row and vertical column stay in bounds.
- Exactly `offset` gray cells are written at `(0..offset-1, 9)` in the input. These are markers only and are absent from the output.
- The output is otherwise black except the shifted cross.

## Reference Notes

The ARC-DSL solver removes gray cells, finds the least nonzero/non-gray color as the cross color, counts gray marker cells for the offset, identifies the cross center by the colored cell whose diagonal neighborhood has four cells of that color, shifts that center by `offset * DOWN_LEFT`, and draws the vertical and horizontal frontiers through the shifted center on a blank canvas.

The Code Golf 2025 solution is a compact rotation trick. It infers the offset from repeated top rows caused by the gray right-column markers, rotates rows by that amount, and rotates each row left by that amount, which has the same effect on this cross-only structure.
