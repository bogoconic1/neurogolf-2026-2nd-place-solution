# task336 Semantics

## Sources

- Current champion builder: `solutions_py/task336.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task336.json`
- ARC-GEN task id: `d4f3cd78`
- ARC-DSL task id: `d4f3cd78`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d4f3cd78.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d4f3cd78.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task336.py`

## Pattern

The input is a 10x10 grid containing a gray `5` container/wall shape on black background. In canonical orientation it is a rectangular open container spanning columns 2..7 with vertical sides at columns 2 and 7, a full near border, and a far border with a single missing cell at the center coordinate 5. The container depth is 5 or 6 and may be offset from the edge by a gap. The whole construction is then rotated/gravity-oriented to one of the four sides.

The output preserves the gray walls, fills the container interior with cyan `8`, and draws a cyan ray out through the missing center gap away from the container until the grid boundary. Background elsewhere stays black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    gray = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 5]
    out = [row[:] for row in grid]
    rs = [r for r, _ in gray]
    cs = [c for _, c in gray]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)

    # Fill the bounding-box interior not occupied by gray.
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if out[r][c] == 0:
                out[r][c] = 8

    # The missing side cell is the black cell on the bbox border centered on
    # one side. Shoot away from the box through that gap.
    candidates = []
    for c in range(c0 + 1, c1):
        if grid[r0][c] == 0:
            candidates.append((r0, c, -1, 0))
        if grid[r1][c] == 0:
            candidates.append((r1, c, 1, 0))
    for r in range(r0 + 1, r1):
        if grid[r][c0] == 0:
            candidates.append((r, c0, 0, -1))
        if grid[r][c1] == 0:
            candidates.append((r, c1, 0, 1))

    if candidates:
        r, c, dr, dc = candidates[0]
        while 0 <= r < h and 0 <= c < w:
            if out[r][c] == 0:
                out[r][c] = 8
            r += dr
            c += dc
    return out
```

## Generator Constraints

- Grid size is fixed at 10x10.
- Container depth is 5 or 6.
- Gap from the adhered side is sampled so the object fits with at least two cells beyond the far border.
- Gravity/orientation is one of four rotations/reflections supplied by `common.apply_gravity`.
- Gray walls use color 5; generated cyan output uses color 8.
- The canonical missing border cell is at coordinate 5 along the container width, and the ray always follows that central opening outward.

## Reference Notes

ARC-DSL fills `delta(ofcolor(I, FIVE))` with cyan, then computes the gray bounding box, finds the box cells missing from the gray object, and shoots from the first missing cell in the direction from the box toward that missing side. Code Golf expresses the same rule with a compact recursive row/column scan.
