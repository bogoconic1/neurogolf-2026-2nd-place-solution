# task240 Semantics

## Sources

- Current champion builder: `solutions_py/task240.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task240.json`
- ARC-GEN task id: `9d9215db`
- ARC-DSL task id: `9d9215db`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_9d9215db.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_9d9215db.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task240.py`

## Pattern

The input is a 19x19 black grid with a sparse colored seed pattern in one corner, possibly horizontally and/or vertically flipped. Seed coordinates are odd positions from a small 2x2 to 4x4 canonical bitmap. Output mirrors every seed to all four symmetric corners. When a diagonal canonical seed has the paired nextdoor color immediately to its right and below, the output also draws a dotted square frame in that nextdoor color along the four symmetric sides, using odd grid positions. The output is still on the 19x19 odd-position lattice and is padded by the NeuroGolf wrapper to 30x30.

## Readable Python Solver

```python
def solve(grid):
    n = 19
    pts = [(r, c, grid[r][c]) for r in range(n) for c in range(n) if grid[r][c]]
    top = min(r for r, _, _ in pts) < n // 2
    left = min(c for _, c, _ in pts) < n // 2

    def to_canon(r, c):
        return (r if top else n - 1 - r, c if left else n - 1 - c)

    def from_canon(r, c):
        return (r if top else n - 1 - r, c if left else n - 1 - c)

    seeds = {}
    for r, c, color in pts:
        seeds[to_canon(r, c)] = color

    out = [[0 for _ in range(n)] for _ in range(n)]

    def paint_canon(r, c, color):
        for rr, cc in ((r, c), (r, n - 1 - c), (n - 1 - r, c), (n - 1 - r, n - 1 - c)):
            y, x = from_canon(rr, cc)
            out[y][x] = color

    for (r, c), color in seeds.items():
        paint_canon(r, c, color)

    for d in (1, 3, 5):
        side = seeds.get((d, d + 2))
        if not side or seeds.get((d + 2, d)) != side:
            continue
        hi = n - 1 - d
        for x in range(d + 2, hi, 2):
            paint_canon(d, x, side)
            paint_canon(hi, x, side)
        for y in range(d + 2, hi, 2):
            paint_canon(y, d, side)
            paint_canon(y, hi, side)
    return out
```

## Generator Constraints

The generated size is fixed at 19. The canonical bitmap length is 2..4. Colors are sampled from 1..9. Diagonal cells are usually present, with the final diagonal sometimes skipped. A nextdoor color may be added symmetrically at `(i,i+1)` and `(i+1,i)`, but the generator avoids three colors in a row. The whole input and output can be independently flipped horizontally and vertically. No nonzero cells appear off the odd lattice in the input.

## Reference Notes

ARC-DSL constructs all rotations, chooses the orientation whose top-left half has the most colors, mirrors it, identifies four-cell objects, draws the dotted connecting lines, then merges rotations back. The Code Golf 2025 solution is a compact fixed 19x19 iterative expression matching the same odd-lattice mirror/frame behavior.
