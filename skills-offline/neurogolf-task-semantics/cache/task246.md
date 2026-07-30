# task246 Semantics

## Sources

- Current champion builder: `solutions_py/task246.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task246.json`
- ARC-GEN task id: `a2fd1cf0`
- ARC-DSL task id: `a2fd1cf0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a2fd1cf0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a2fd1cf0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task246.py`

## Pattern

The input is a rectangular grid, width and height in the 10..20 range for
random ARC-GEN samples, padded to the NeuroGolf 30x30 tensor. It contains one
red pixel (color 2) and one green pixel (color 3) on black background. The
output keeps those endpoints and draws a cyan Manhattan connector (color 8): a
horizontal segment on the red pixel's row between the red column and the green
column, and a vertical segment on the green pixel's column between the red row
and the green row. The endpoint cells remain red/green because the ARC-DSL
solver uses `underfill`, so cyan is written only into black cells along the two
segments.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    red = green = None
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                red = (r, c)
            elif grid[r][c] == 3:
                green = (r, c)
    rr, rc = red
    gr, gc = green

    for c in range(min(rc, gc), max(rc, gc) + 1):
        if out[rr][c] == 0:
            out[rr][c] = 8
    for r in range(min(rr, gr), max(rr, gr) + 1):
        if out[r][gc] == 0:
            out[r][gc] = 8
    return out
```

## Generator Constraints

- Random grids have width and height independently sampled from 10..20.
- Red/green rows are two distinct samples from `range(1, height - 2)`.
- Red/green columns are two distinct samples from `range(1, width - 2)`.
- Endpoints are strictly interior, not on the padded border.
- There is exactly one red pixel and exactly one green pixel.
- The output is produced by `common.hpwl(..., black, red, green, cyan)`, matching
  the horizontal-plus-vertical path described above.
- The official ARC train/test examples include sizes up to 16x15 in the bundled
  JSON; NeuroGolf still requires dense `[1,10,30,30]` tensor output.

## Reference Notes

ARC-DSL computes red row/column and green row/column, connects `(red_row, green_col)` to `(green_row, green_col)`, connects `(red_row, red_col)` to `(red_row, green_col)`, combines those coordinates, and underfills color 8 into input. The Code Golf solution recursively rotates/pops the grid while drawing cyan when a row/column contains the endpoint colors; this is another compact view of the same orthogonal connector.
