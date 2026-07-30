# task046 Semantics

## Sources

- Current champion builder: `solutions_py/task046.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task046.json`
- ARC-GEN task id: `234bbc79`
- ARC-DSL task id: `234bbc79`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_234bbc79.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_234bbc79.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task046.py`

## Pattern

The input is a 3-row drawing made from several short colored path segments that
belong to one continuous polyline. Neighboring segments have been separated by
one blank column, vertically shifted independently, and their touching endpoint
pixels are colored gray (`5`). The output removes the inserted separator columns,
restores gray join pixels to the color of their segment, and shifts the pieces
up or down so the nearest gray endpoints reconnect into the original continuous
3-row path.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])

    def neighbors(cell):
        r, c = cell
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0:
                yield nr, nc

    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                cell = stack.pop()
                comp.append(cell)
                for nb in neighbors(cell):
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            comps.append(comp)

    comps.sort(key=lambda comp: min(c for _, c in comp))

    pieces = []
    for comp in comps:
        colors = {grid[r][c] for r, c in comp if grid[r][c] not in (0, 5)}
        color = next(iter(colors)) if colors else 5
        min_c = min(c for _, c in comp)
        cells = {(r, c - min_c, color) for r, c in comp}
        left_gray = [r for r, c in comp if c == min_c and grid[r][c] == 5]
        max_c = max(c for _, c in comp)
        right_gray = [r for r, c in comp if c == max_c and grid[r][c] == 5]
        pieces.append({"cells": cells, "left": left_gray, "right": right_gray})

    placed = []
    offset = 0
    x = 0
    for i, piece in enumerate(pieces):
        if i:
            prev = pieces[i - 1]
            prev_right = prev["right"] or [r for r, c, _ in prev["cells"] if c == max(c0 for _, c0, _ in prev["cells"])]
            cur_left = piece["left"] or [r for r, c, _ in piece["cells"] if c == 0]
            best = min((abs((r2 + offset) - r1), r1 - r2) for r1 in prev_right for r2 in cur_left)[1]
            offset += best
        for r, c, color in piece["cells"]:
            placed.append((r + offset, c + x, color))
        x += 1 + max(c for _, c, _ in piece["cells"])

    out = [[0 for _ in range(x)] for _ in range(3)]
    for r, c, color in placed:
        if 0 <= r < 3:
            out[r][c] = color
    return out
```

This solver is intentionally readable rather than golfed; it mirrors the
reference idea of finding foreground objects, recoloring each object away from
gray, ordering objects left-to-right, and shifting each next object to the
closest compatible join.

## Generator Constraints

ARC-GEN always uses height `3`. It creates either three or four segments. Each
segment width is `2..4`, so the final output width is the sum of those widths.
The input width is output width plus one inserted separator column between every
pair of segments, plus an optional extra blank column at the end. Segment colors
are non-gray random colors and may repeat. Within each segment the path can run
horizontally and may make a vertical turn while staying within the three rows.
Every segment after the first receives an independent vertical offset that keeps
it in bounds. Boundary pixels adjacent to another segment are colored gray in
the input; the output uses the segment's true color at those locations.

## Reference Notes

The ARC-DSL solver forms foreground objects, recolors gray pixels in each object using the other color from the object's palette, orders pieces by left edge, and iteratively shifts the next piece so its left/right gray endpoint is nearest to the already placed path endpoint. It then paints the assembled object on a 3-row canvas whose width is the assembled object's width. The Code Golf 2025 solution is a compact row-window expression that removes gray separator columns and recenters the three-row slices around gray joins; it confirms that the task is only about 3-row segment reassembly, not arbitrary object recognition.
