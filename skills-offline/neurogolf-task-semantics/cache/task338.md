# task338 Semantics

## Sources

- Current champion builder: `solutions_py/task338.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task338.json`
- ARC-GEN task id: `d5d6de2d`
- ARC-DSL task id: `d5d6de2d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d5d6de2d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d5d6de2d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task338.py`

## Pattern

The input is a square ARC grid of size 10, 15, 20, or 25, encoded in the
standard NeuroGolf `[1,10,30,30]` one-hot tensor. The visible colors are black
`0` and red `2`. The red cells form one or more non-overlapping axis-aligned
rectangle borders, separated by at least one cell of background.

The output removes every red border cell and writes green `3` into the strict
interior of each rectangle. Background stays black. Rectangles with width 2 or
height 2 have no strict interior, so they disappear. There is no tie-breaking
or color choice: all filled cells use color 3.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    seen = [[False for _ in range(w)] for _ in range(h)]

    for r0 in range(h):
        for c0 in range(w):
            if grid[r0][c0] != 2 or seen[r0][c0]:
                continue

            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr][cc]:
                        if grid[rr][cc] == 2:
                            seen[rr][cc] = True
                            stack.append((rr, cc))

            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            top, bottom = min(rs), max(rs)
            left, right = min(cs), max(cs)

            for r in range(top + 1, bottom):
                for c in range(left + 1, right):
                    out[r][c] = 3

    return out
```

## Generator Constraints

- Grid size is `5 * mult` where `mult` is 2, 3, 4, or 5, so the real grid is at most 25x25 inside the 30x30 NeuroGolf tensor.
- Number of boxes is `mult - 1` or `mult`.
- Each box width and height is independently sampled from 2 through 9.
- Boxes are placed randomly with a one-cell non-overlap margin.
- The generator first paints each whole rectangle red, then resets the strict interior to black in the input and green in the output.
- Width-2 or height-2 boxes have no interior and therefore contribute no green cells. Square boxes are possible; 2x2 squares vanish because they have no interior, while larger square borders still fill their centers.

## Reference Notes

ARC-GEN is the clearest source: it directly constructs red rectangle borders and green interiors. ARC-DSL uses `objects`, drops objects that satisfy its `square` predicate, takes `backdrop(inbox(...))`, replaces red with black, and fills the collected interiors green. This matches the generated examples because the only square-like red object that should disappear without fill is the 2x2 solid case with no interior. The Code Golf solution is a row-wise parity scan: it tracks side crossings across each row and emits green when the scan is inside a border pair.
