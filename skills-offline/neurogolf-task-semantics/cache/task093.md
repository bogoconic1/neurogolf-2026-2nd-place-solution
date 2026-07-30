# task093 Semantics

## Sources

- Current champion builder: `solutions_py/task093.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task093.json`
- ARC-GEN task id: `4093f84a`
- ARC-DSL task id: `4093f84a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4093f84a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4093f84a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task093.py`

## Pattern

The grid is 14x14. A solid gray `5` horizon band of thickness `2..5` spans the
whole grid in one orientation. Sparse pixels of one non-gray color appear only
on the two sides of the band. The output keeps the gray band, converts all
colored pixels to gray, and compacts them toward the nearest side of the band
within their row/column line. Multiple pixels in the same line stack outward
from the band. The whole pattern may be vertically flipped and/or transposed, so
the horizon can appear from either side and in either horizontal/vertical
orientation.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    out = [[0 for _ in row] for row in grid]

    gray = [(r, c) for r in range(n) for c in range(n) if grid[r][c] == 5]
    rows = {r for r, c in gray}
    cols = {c for r, c in gray}
    horizontal = len(rows) < len(cols)

    work = [row[:] for row in grid]
    transposed = False
    if not horizontal:
        work = [list(row) for row in zip(*work)]
        transposed = True

    gray_rows = [r for r in range(n) if all(work[r][c] == 5 for c in range(n))]
    top = min(gray_rows)
    bottom = max(gray_rows)

    out = [[0] * n for _ in range(n)]
    for r in range(top, bottom + 1):
        for c in range(n):
            out[r][c] = 5

    for c in range(n):
        above = sum(work[r][c] not in (0, 5) for r in range(top))
        below = sum(work[r][c] not in (0, 5) for r in range(bottom + 1, n))
        for k in range(above):
            out[top - 1 - k][c] = 5
        for k in range(below):
            out[bottom + 1 + k][c] = 5

    if transposed:
        out = [list(row) for row in zip(*out)]
    return out
```

## Generator Constraints

ARC-GEN uses size `14`. The canonical band is horizontal at rows `5..5+thick-1`
with `thick` in `2..5`. Colored pixels are sampled sparsely outside the band,
with at least one colored pixel, and all use one random non-gray color. The
instance may be flipped vertically and may be transposed, so the solver must not
assume the band is always horizontal or that gravity is always downward.

## Reference Notes

The ARC-DSL solver replaces the least frequent non-background color with gray, canonicalizes orientation using the gray object, splits the canonical grid into two halves around the band, orders one half normally and the other inverted, then mirrors back if needed. The Code Golf solution uses repeated transpose/rotation and regex replacement to move non-gray pixels into gray stacks. All references agree that the output contains only black and gray.
