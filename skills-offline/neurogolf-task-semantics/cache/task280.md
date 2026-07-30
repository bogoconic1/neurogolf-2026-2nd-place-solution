# task280 Semantics

## Sources

- Current champion builder: `solutions_py/task280.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task280.json`
- ARC-GEN task id: `b527c5c6`
- ARC-DSL task id: `b527c5c6`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b527c5c6.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b527c5c6.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task280.py`

## Pattern

The grid contains two green (`3`) rectangular objects on a black background.
Each rectangle contains one red (`2`) marker on one edge. One rectangle is
tall/narrow and has a marker on its left or right edge; the other is
wide/short and has a marker on its top edge after the generator's optional
flip/transpose branches, so in canonical coordinates the two markers emit
one horizontal and one vertical ray.

For each object, determine which edge contains the red marker. From that red
cell, shoot an axis-aligned ray outward away from the object edge until the
grid boundary. The center ray is red. Around that center ray, add parallel
green rays with offset radius `min(object_height, object_width) - 1`, producing
a thick beam whose width matches the object's smaller dimension. The original
rectangles remain. Red center rays take priority over green beam fill.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    seen = [[False] * w for _ in range(h)]
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] != 0:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            objects.append(cells)

    red_rays = []
    green_rays = []
    for cells in objects:
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        red = [(r, c) for r, c in cells if grid[r][c] == 2]
        if not red:
            continue
        rr, cc = red[0]
        dr = -1 if rr == r0 else (1 if rr == r1 else 0)
        dc = -1 if cc == c0 else (1 if cc == c1 else 0)
        radius = min(r1 - r0 + 1, c1 - c0 + 1) - 1

        ray = []
        r, c = rr, cc
        while 0 <= r < h and 0 <= c < w:
            ray.append((r, c))
            r += dr
            c += dc
        red_rays.extend(ray)
        if dr:
            for off in range(-radius, radius + 1):
                green_rays.extend((r, c + off) for r, c in ray if 0 <= c + off < w)
        else:
            for off in range(-radius, radius + 1):
                green_rays.extend((r + off, c) for r, c in ray if 0 <= r + off < h)

    for r, c in green_rays:
        if out[r][c] == 0:
            out[r][c] = 3
    for r, c in red_rays:
        out[r][c] = 2
    return out
```

## Generator Constraints

ARC-GEN uses square grids of size 10 or 20, with optional vertical flip and
optional transpose. There are exactly two rectangles, both filled green, and
each has exactly one red marker. Before flip/transpose, rectangle 0 is
tall/narrow, rectangle 1 is wide/short, rectangle 0 is left of rectangle 1, and
the markers are placed far enough from corners to support the full beam width.
The tall rectangle can have its red marker on either the left or right edge;
the right-edge case is the generator's `defect` branch. The wide rectangle's
marker is on its top edge. All examples use only colors 0, 2, and 3.

## Reference Notes

ARC-DSL derives the outward direction from whether the red marker lies on the object's top, bottom, left, or right boundary. It fills the red center rays first and then underfills the green parallel rays, preserving red priority. The beam radius is `min(shape)-1`. Code Golf 2025 encodes the same repeated rotate/transpose logic compactly: identify edge markers, extend the red/green beam in one orientation, then rotate through orientations.
