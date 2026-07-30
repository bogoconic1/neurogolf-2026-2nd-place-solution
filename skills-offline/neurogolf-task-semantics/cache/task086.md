# task086 Semantics

## Sources

- Current champion builder: `solutions_py/task086.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task086.json`
- ARC-GEN task id: `3befdf3e`
- ARC-DSL task id: `3befdf3e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3befdf3e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3befdf3e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task086.py`

## Pattern

The input is a black square grid containing one or two non-overlapping flower centers. Each center is a solid inner square of color A with side length 1 or 2, surrounded by a one-cell-thick square frame of color B. Thus each input object is either a 3x3 framed square with a 1x1 center, or a 4x4 framed square with a 2x2 center.

The output swaps the two object colors inside each framed square: the original B frame becomes color A, and the original A center becomes color B. It also extends color B outward around the object by the center length plus one cells, forming a larger plus-like flower. The extension is the larger bounding square around the object, except that the four corner blocks outside the original frame's row band and column band remain black. Multiple objects are processed independently; the generator prevents overlap.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    colors = sorted({v for row in grid for v in row if v})
    if len(colors) != 2:
        return out
    counts = {c: sum(v == c for row in grid for v in row) for c in colors}
    inner = min(colors, key=lambda c: counts[c])
    outer = max(colors, key=lambda c: counts[c])
    if counts[colors[0]] > counts[colors[1]]:
        inner, outer = colors[1], colors[0]

    seen = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or seen[r][c]:
                continue
            cells = []
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] != 0:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            rs = [rr for rr, _ in cells]
            cs = [cc for _, cc in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            frame_side = r1 - r0 + 1
            length = frame_side - 2

            # First add the outer-color flower extent, skipping the four corner
            # blocks outside the original frame row/column bands.
            for rr in range(r0 - length, r1 + length + 1):
                for cc in range(c0 - length, c1 + length + 1):
                    if not (0 <= rr < h and 0 <= cc < w):
                        continue
                    outside_rows = rr < r0 or rr > r1
                    outside_cols = cc < c0 or cc > c1
                    if outside_rows and outside_cols:
                        continue
                    out[rr][cc] = outer

            # Then swap colors in the original framed square.
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    out[rr][cc] = inner
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    out[rr][cc] = outer
    return out
```

## Generator Constraints

The grid is square with size 10, 11, or 12. The generator tries to place two objects, but skips a sampled object if its enlarged flower extent would overlap an existing one, so examples may contain one or two objects. Each object has center side length 1 or 2. The anchor row and column leave enough margin for the full expansion: `row` and `col` are sampled from `length + 1` through `size - 2 * length - 2` inclusive. The two non-background colors are random and shared by all objects in the grid. The inner color is always the less frequent color in the input.

## Reference Notes

ARC-DSL identifies non-background objects, finds the least frequent color as the inner color, swaps the two object colors, then computes each object's expanded outbox and removes the four expanded corner regions. The Code Golf solution encodes the same recursive local relation: colors are swapped in the center/frame and non-corner extension cells are filled.
