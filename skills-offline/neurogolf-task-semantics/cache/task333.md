# task333 Semantics

## Sources

- Current champion builder: `solutions_py/task333.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task333.json`
- ARC-GEN task id: `d43fd935`
- ARC-DSL task id: `d43fd935`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d43fd935.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d43fd935.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task333.py`

## Pattern

The input and output are 10x10 grids. A fixed 2x2 green box (`3`) appears away from the border. Other foreground cells are isolated singleton pixels in one or two non-green colors. For every non-green singleton that lies on one of the two rows occupied by the green box, draw a horizontal line from that singleton toward the nearest side of the green box, stopping before/at the green box. For every non-green singleton that lies on one of the two columns occupied by the green box, draw a vertical line from that singleton toward the nearest side of the green box. The line uses the singleton's own color. Preserve the green box and all original pixels.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    greens = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 3]
    br = min(r for r, c in greens)
    bc = min(c for r, c in greens)
    box_rows = {br, br + 1}
    box_cols = {bc, bc + 1}
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color in (0, 3):
                continue
            if r in box_rows:
                step = -1 if c > bc else 1
                cc = c
                while out[r][cc + step] != 3:
                    cc += step
                    out[r][cc] = color
            if c in box_cols:
                step = -1 if r > br else 1
                rr = r
                while out[rr + step][c] != 3:
                    rr += step
                    out[rr][c] = color
    return out
```

## Generator Constraints

ARC-GEN uses a 10x10 square grid. The green 2x2 box top-left corner is sampled with `boxrow, boxcol` in `[2, size-3]`, so the box has at least two cells of margin on all sides. It chooses one or two non-green colors. Random singleton pixels may be anywhere, but at most one left/right/up/down aligned guide per generated color is selected by flags. Extra off-axis singleton pixels remain unchanged. The generator rejects samples unless every cell in the one-cell halo around the green 2x2 box, excluding black/green, is absent; this guarantees no adjacent colored pixel immediately touching the box and leaves a clear line segment to fill.

## Reference Notes

The ARC-DSL solver finds singleton objects, filters those whose center shares a row or column with the green object, computes the gravity vector from each singleton toward the green box, connects the singleton center to that destination, recolors the path with the singleton color, and paints the paths over the input. The Code Golf solution is a compact recursive/transposed scan that propagates colors along rows and columns toward the green block.
