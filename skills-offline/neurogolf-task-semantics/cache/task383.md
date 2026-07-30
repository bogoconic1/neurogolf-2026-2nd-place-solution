# task383 Semantics

## Sources

- Current champion builder: `solutions_py/task383.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task383.json`
- ARC-GEN task id: `f1cefba8`
- ARC-DSL task id: `f1cefba8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f1cefba8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f1cefba8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task383.py`

## Pattern

The input is a black canvas containing one solid nonzero rectangle with two colors.  The outer two-cell-thick frame uses an outer color `A`; the inner rectangle uses an inner color `B`.  A small number of `B` cells appear as "barnacles" in the frame:

- a `B` cell in the top or bottom frame, aligned with an inner column, marks a vertical guide line at that column
- a `B` cell in the left or right frame, aligned with an inner row, marks a horizontal guide line at that row

The output has the same size as the input.  It draws every marked guide line through the whole canvas in color `B`, then restores the original rectangle geometry: the outer frame is color `A`, and the inner rectangle is color `B` except where a marked horizontal or vertical guide crosses it, where those cells become color `A`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    pts = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != 0]
    r0, r1 = min(r for r, _ in pts), max(r for r, _ in pts) + 1
    c0, c1 = min(c for _, c in pts), max(c for _, c in pts) + 1
    outer = grid[r0][c0]
    inner = next(v for row in grid for v in row if v not in (0, outer))

    marked_rows = set()
    marked_cols = set()
    for r in range(r0, r1):
        for c in range(c0, c1):
            if grid[r][c] != inner:
                continue
            in_inner_rows = r0 + 2 <= r < r1 - 2
            in_inner_cols = c0 + 2 <= c < c1 - 2
            if in_inner_rows and in_inner_cols:
                continue
            if in_inner_rows:
                marked_rows.add(r)
            if in_inner_cols:
                marked_cols.add(c)

    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in marked_rows:
        for c in range(w):
            out[r][c] = inner
    for c in marked_cols:
        for r in range(h):
            out[r][c] = inner

    for r in range(r0, r1):
        for c in range(c0, c1):
            out[r][c] = outer
    for r in range(r0 + 2, r1 - 2):
        for c in range(c0 + 2, c1 - 2):
            out[r][c] = outer if r in marked_rows or c in marked_cols else inner
    return out
```

## Generator Constraints

The generated rectangle width and height are each 8..16.  The full canvas is the rectangle plus 4..8 extra rows/columns, and the rectangle is placed away from the canvas border.  The frame is exactly two cells thick.  There are 2 or 3 barnacles.  A top/bottom barnacle chooses a column within the inner width and places a `B` cell in the top or bottom two-cell frame; a left/right barnacle chooses a row within the inner height and places a `B` cell in the left or right two-cell frame.  Colors are two distinct random nonzero ARC colors.  The input has no other nonzero objects.

## Reference Notes

The ARC-DSL solution finds the single nonzero object, trims the two-cell frame to reason about the interior, determines the two nonzero colors, detects the frame barnacles, builds horizontal and vertical frontiers from their aligned rows/columns, and recolors their intersection with the original black background as inner color while restoring the rectangle colors. The compact Code Golf solution performs repeated rotations/transposes and palette rewriting; it agrees with the generator that the transformation is row/column guide extension from frame markers.
