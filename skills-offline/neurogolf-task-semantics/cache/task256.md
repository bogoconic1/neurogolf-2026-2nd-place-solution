# task256 Semantics

## Sources

- Current champion builder: `solutions_py/task256.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task256.json`
- ARC-GEN task id: `a65b410d`
- ARC-DSL task id: `a65b410d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a65b410d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a65b410d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task256.py`

## Pattern

The input is a blank grid with a red (`2`) horizontal line segment starting at the left edge. If the red segment has length `length` and is on row `row`, then `triangle = row + length`. The output is the same grid with a left-aligned triangular prefix filled over rows `0..triangle-1`.

For each output row `r < triangle`, columns `0..triangle-r-1` are colored. Rows above the red segment (`r < row`) are green (`3`), the red row keeps the red segment (`2`) of length `length`, and rows below it (`row < r < triangle`) are blue (`1`). Everything outside that left prefix and every row at or below `triangle` remains black (`0`).

The visible geometry is a right triangle whose upper-left corner is at `(0,0)` and whose descending diagonal passes through the upper-right end of the red segment at `(row, length-1)`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    red_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 2]
    if not red_cells:
        return [row[:] for row in grid]
    red_row = red_cells[0][0]
    length = max(c for r, c in red_cells if r == red_row) + 1
    triangle = red_row + length

    out = [row[:] for row in grid]
    for r in range(min(triangle, h)):
        prefix = min(triangle - r, w)
        if r < red_row:
            color = 3
        elif r == red_row:
            color = 2
        else:
            color = 1
        for c in range(prefix):
            out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN chooses `triangle` in `[3, 9]`, then sets `width = triangle + randint(1,4)` and `height = triangle + randint(1,4)`, so the triangle always fits with at least one blank row and column margin. The red segment length is `length in [2, triangle-1]`, and `row = triangle - length`, so the red row is never row 0 and never below the triangle. Input contains only black and red. Output contains only black, blue (`1`), red (`2`), and green (`3`). The red segment always starts at column 0 and is contiguous.

## Reference Notes

The ARC-DSL solver finds the red cells, takes the upper-right corner of the red segment, shoots an up-right ray and a down-left ray from that corner, and underfills leftward from those rays: green above the red line and blue below it. This confirms the diagonal boundary is determined by the right end of the red segment, not by the full canvas size.

The Code Golf 2025 solution computes `w = red_sum // 2` as the red length and `i = w + red_row` as the triangle size, then walks rows while decreasing `i`. The compact color expression switches from green to red to blue based on whether the current prefix length is above, equal to, or below the red length.
