# task064 Semantics

## Sources

- Current champion builder: `solutions_py/task064.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task064.json`
- ARC-GEN task id: `2c608aff`
- ARC-DSL task id: `2c608aff`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_2c608aff.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_2c608aff.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task064.py`

## Pattern

The input has a dominant background color, one solid rectangle of a second color, and sparse dots of a third color. For every dot that is horizontally or vertically aligned with the rectangle, draw a straight line of the dot color through background cells until it reaches the rectangle. Dots that are diagonal from the rectangle do not extend. Existing rectangle and dot cells remain unchanged.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    colors = {}
    for row in grid:
        for v in row:
            colors[v] = colors.get(v, 0) + 1
    bg = max(colors, key=colors.get)
    dot_color = min(colors, key=colors.get)
    box_color = next(c for c in colors if c not in (bg, dot_color))
    box = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == box_color]
    r0, r1 = min(r for r, _ in box), max(r for r, _ in box)
    c0, c1 = min(c for _, c in box), max(c for _, c in box)
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != dot_color:
                continue
            if r0 <= r <= r1:
                step = 1 if c < c0 else -1 if c > c1 else 0
                if step:
                    cc = c + step
                    while out[r][cc] != box_color:
                        if out[r][cc] == bg:
                            out[r][cc] = dot_color
                        cc += step
            elif c0 <= c <= c1:
                step = 1 if r < r0 else -1 if r > r1 else 0
                if step:
                    rr = r + step
                    while out[rr][c] != box_color:
                        if out[rr][c] == bg:
                            out[rr][c] = dot_color
                        rr += step
    return out
```

## Generator Constraints

Width and height are sampled independently from `8..24`. The rectangle width is `3..width//2`, height is `3..height//2`, and its top-left corner leaves at least one-cell margin from every border. Sparse dot pixels are sampled with density about `0.02` but rejected if they are inside or adjacent to the rectangle envelope. Colors are three distinct random colors: rectangle, dot, and background. Dot lines are drawn only for dots sharing a row with the rectangle row span or a column with the rectangle column span; diagonal dots are unchanged.

## Reference Notes

The ARC-DSL solver uses `leastcolor` as the dot color, finds the largest object as the rectangle, connects every rectangle cell to every dot-color cell, filters those connections to horizontal/vertical lines, and underfills them with the dot color. The Code Golf solution is a compact recursive/transpose-oriented version of the same idea: identify the rare dot color and fill aligned background cells up to the rectangle.
