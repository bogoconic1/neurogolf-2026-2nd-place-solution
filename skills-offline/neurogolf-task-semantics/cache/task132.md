# task132 Semantics

## Sources

- Current champion builder: `solutions_py/task132.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task132.json`
- ARC-GEN task id: `56ff96f3`
- ARC-DSL task id: `56ff96f3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_56ff96f3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_56ff96f3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task132.py`

## Pattern

The input contains one or two colored rectangle markers on a black background. For each foreground color, exactly two cells of that color mark opposite corners of an axis-aligned rectangle. The marked corners may be top-left plus bottom-right, or top-right plus bottom-left. The output fills the entire inclusive bounding box spanned by the two cells with that color. Multiple rectangles use distinct colors and are generated with at least one cell of separation, so their filled boxes do not overlap.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    colors = sorted({v for row in grid for v in row if v != 0})
    for color in colors:
        cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == color]
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN samples grid width and height from `6..15`. It creates one or two boxes. Each box width is `2..width-1`, height is `2..height-1`, and the top-left placement leaves at least one unused row/column beyond the bottom/right edge. Boxes are retried until the requested count is placed without overlap using a one-cell separation margin. Each box receives a random foreground color. A per-box flip bit chooses which diagonal pair is shown in the input: top-left/bottom-right when false, top-right/bottom-left when true. The output fills every cell inside each marked bounding box.

## Reference Notes

The ARC-DSL solver partitions foreground objects, takes each object's `backdrop` (its bounding box), recolors that box with the object's color, and paints all boxes onto the input. The Code Golf solution repeatedly propagates colors across rows and columns until each marked rectangle is filled. All three references agree that the only semantic state needed per color is the bounding row and column range of its two marked cells.
