# task349 Semantics

## Sources

- Current champion builder: `solutions_py/task349.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task349.json`
- ARC-GEN task id: `db93a21d`
- ARC-DSL task id: `db93a21d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_db93a21d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_db93a21d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task349.py`

## Pattern

The input contains one or more maroon (`9`) square blocks on a black background.
Each block has even side length `2*r`, where `r` is 1, 2, or 3. The output keeps
the maroon blocks, draws a blue (`1`) vertical beam downward from every maroon
cell to the bottom of the grid, and draws a green (`3`) halo/outbox of thickness
`r` around each maroon block. Maroon overwrites the halo in the center, and blue
is underfilled so the maroon source cells stay maroon.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    seen = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 9 or (r, c) in seen:
                continue

            stack = [(r, c)]
            cells = []
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] == 9:
                        seen.add((nr, nc))
                        stack.append((nr, nc))

            r0, r1 = min(x for x, _ in cells), max(x for x, _ in cells)
            c0, c1 = min(y for _, y in cells), max(y for _, y in cells)
            radius = (c1 - c0 + 1) // 2

            for rr in range(r0 - radius, r1 + radius + 1):
                for cc in range(c0 - radius, c1 + radius + 1):
                    if 0 <= rr < h and 0 <= cc < w:
                        out[rr][cc] = 3
            for rr, cc in cells:
                for br in range(rr + 1, h):
                    if out[br][cc] == 0:
                        out[br][cc] = 1
            for rr, cc in cells:
                out[rr][cc] = 9
    return out
```

## Generator Constraints

ARC-GEN chooses a square grid of size `5*factor`, with `factor` from 2 to 6, so
sizes are `10,15,20,25,30`. It places up to `factor` non-overlapping maroon
squares. A radius is chosen from `1..factor-1`, and the center block is `2*r` by
`2*r`. Placement constraints leave enough horizontal room for the block and
reject overlapping halo/beam regions using padded bounding tests. The tuples are
sorted by row. Halos and beams can clip at grid borders via `common.draw`.

## Reference Notes

The ARC-DSL solver identifies maroon objects, shoots blue downward from maroon cells, and fills green using one, two, or three repeated `outbox` layers depending on half-width/radius. The Code Golf 2025 solution encodes the same iterative scan/propagation rule compactly. The references agree that object size controls halo thickness and that maroon centers remain maroon.
