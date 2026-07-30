# task204 Semantics

## Sources

- Current champion builder: `solutions_py/task204.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task204.json`
- ARC-GEN task id: `868de0fa`
- ARC-DSL task id: `868de0fa`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_868de0fa.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_868de0fa.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task204.py`

## Pattern

The input is a square grid containing several non-overlapping hollow blue square outlines. Blue is color `1`; the background is black. The output preserves every blue outline and fills each square interior according to the square side length: even side length becomes red (`2`), odd side length becomes orange (`7`). Output size equals input size.

There is no recoloring of the outline. Only cells strictly inside each square are changed. Squares are axis-aligned and can have side length from 3 to 10, so every square has at least a 1x1 interior.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    seen = set()

    def neighbors(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                yield rr, cc

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 1 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors(rr, cc):
                    if grid[nr][nc] == 1 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))

            rs = [rr for rr, _ in comp]
            cs = [cc for _, cc in comp]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            side = r1 - r0 + 1
            if side != c1 - c0 + 1:
                continue
            fill = 2 if side % 2 == 0 else 7
            for rr in range(r0 + 1, r1):
                for cc in range(c0 + 1, c1):
                    out[rr][cc] = fill
    return out
```

## Generator Constraints

- Grid size is square, sampled from `10..20`.
- Number of boxes is sampled from `size // 4 .. size // 3`.
- Each box side length is independently sampled from `3..10`.
- Each box top-left is sampled so the full square fits inside the grid.
- Boxes are rejected and resampled until their square bounding boxes do not overlap with a one-cell separation margin.
- Inputs contain only black background and blue square outlines. Outputs keep the outlines and fill interiors with color `2` for even side length and color `7` for odd side length.

## Reference Notes

ARC-DSL extracts objects, filters square objects, splits them by even height, then fills even squares with `TWO` and odd squares with `SEVEN`. This confirms that parity of side length is the only branch and that object outlines remain present.

The Code Golf solution uses a regex over rows to find blue outline runs and repeats a fill value based on the detected interior width parity, matching the same even/odd square-interior rule.

No disagreement was found among ARC-GEN, ARC-DSL, and Code Golf 2025.
