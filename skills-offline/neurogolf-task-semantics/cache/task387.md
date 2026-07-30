# task387 Semantics

## Sources

- Current champion builder: `solutions_py/task387.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task387.json`
- ARC-GEN task id: `f35d900a`
- ARC-DSL task id: `f35d900a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f35d900a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f35d900a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task387.py`

## Pattern

The input is a black rectangular canvas, width and height `14..18`, containing exactly four non-background colored pixels at the corners of an axis-aligned rectangle. The top-left and bottom-right corners have one non-gray color, and the top-right and bottom-left corners have a second non-gray color. The output keeps the same canvas size. Around each corner it draws a `3x3` block in the opposite corner color, then restores the original center pixel. It also adds gray (`5`) markers on the rectangle border at even offsets from each corner, symmetrically from both ends of every side.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    pts = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != 0]
    rows = [r for r, c, color in pts]
    cols = [c for r, c, color in pts]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    colors = sorted({color for r, c, color in pts})

    def other_color(color):
        return colors[1] if color == colors[0] else colors[0]

    for r, c, color in pts:
        fill = other_color(color)
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr][cc] = fill
        out[r][c] = color

    wide = c1 - c0
    tall = r1 - r0
    for dc in range(2, 1 + wide // 2, 2):
        for r in (r0, r1):
            out[r][c0 + dc] = 5
            out[r][c1 - dc] = 5
    for dr in range(2, 1 + tall // 2, 2):
        for c in (c0, c1):
            out[r0 + dr][c] = 5
            out[r1 - dr][c] = 5
    return out
```

## Generator Constraints

ARC-GEN samples width and height independently from `14..18`. The hidden rectangle has width and height `5..11`; its top-left corner is at least one cell from the canvas border and leaves at least two cells of margin beyond the bottom/right corner. There are exactly two nonzero colors and neither is gray (`5`). Only the four rectangle corners are colored in the input. The gray border markers are placed at even offsets `2, 4, ...` up to half the side length, mirrored from both ends; duplicated center-side markers are harmless because they write the same gray value.

## Reference Notes

ARC-DSL finds the four colored singleton objects, recolors each object's `outbox` with the other palette color, paints those corner boxes, computes the bounding box border around the four corner pixels, removes the corner pixels, then fills gray on border cells whose Manhattan distance to the nearest corner has even parity. The Code Golf 2025 solution is a recursive row/column transform but agrees with the same opposite-color corner boxes and parity-gray border rule.
