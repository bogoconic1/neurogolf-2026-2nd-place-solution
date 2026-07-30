# task051 Semantics

## Sources

- Current champion builder: `solutions_py/task051.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task051.json`
- ARC-GEN task id: `25d487eb`
- ARC-DSL task id: `25d487eb`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_25d487eb.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_25d487eb.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task051.py`

## Pattern

The input contains one small triangular laser/arrow object in one color and a single marker cell in a second color at the tip/center of the object. The marker color is the least frequent non-background color in the input. The output keeps the input object and extends a straight beam from the marker in the direction pointing away from the body of the object until the edge of the grid. The beam uses the marker color and only fills background cells; existing object cells remain unchanged. ARC-GEN applies one of four rotations/reflections by `apply_gravity`, so the beam may point up, down, left, or right in the final input orientation.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0) + 1
    beam = min(counts, key=counts.get)
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v]
    mark = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == beam]
    mr = sum(r for r, _ in mark) / len(mark)
    mc = sum(c for _, c in mark) / len(mark)
    cr = sum(r for r, _ in pts) / len(pts)
    cc = sum(c for _, c in pts) / len(pts)
    dr = 0 if abs(cr - mr) < 1e-9 else (1 if cr > mr else -1)
    dc = 0 if abs(cc - mc) < 1e-9 else (1 if cc > mc else -1)
    # The DSL shoot direction points from marker toward the body center; underfill
    # extends along that ray in the oriented grid produced by ARC-GEN.
    out = [row[:] for row in grid]
    r, c = int(round(mr)), int(round(mc))
    while 0 <= r < h and 0 <= c < w:
        if out[r][c] == 0:
            out[r][c] = beam
        r += dr
        c += dc
    return out
```

## Generator Constraints

ARC-GEN samples width and height from `10..20`, triangle depth from `3..4`, a row/column that keeps the unrotated object away from the border, two random non-background colors, and a gravity/orientation value from `0..3`. Before orientation, the body is a filled triangular wedge in `colors[0]`, the tip marker at `(row, col)` is `colors[1]`, and the output fills a vertical beam from `row + depth` to the bottom of the grid in `colors[1]`. `apply_gravity` rotates/reforients both input and output together, so the final task is orientation-invariant.

## Reference Notes

The ARC-DSL solver chooses `leastcolor(I)` as the marker/beam color, finds the marker center, finds the merged non-background object center, computes the vector from marker to object center, shoots a ray from the marker in that direction, and underfills the ray with the marker color. The Code Golf solution rotates the grid repeatedly and uses a regex to fill the oriented `0, marker, 0` pattern, then returns after four rotations.
