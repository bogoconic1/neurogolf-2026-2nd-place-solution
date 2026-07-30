# task253 Semantics

## Sources

- Current champion builder: `solutions_py/task253.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task253.json`
- ARC-GEN task id: `a61ba2ce`
- ARC-DSL task id: `a61ba2ce`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a61ba2ce.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a61ba2ce.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task253.py`

## Pattern

The input is a square grid, generated as 13x13 for ARC-GEN, containing four non-overlapping colored 2x2-corner triominoes on a black background. Each object is one of the four possible 2x2 blocks with exactly one corner missing:

- missing lower-right: cells at upper-left, upper-right, lower-left
- missing lower-left: cells at upper-left, upper-right, lower-right
- missing upper-right: cells at upper-left, lower-left, lower-right
- missing upper-left: cells at upper-right, lower-left, lower-right

The output is a 4x4 grid made from four 2x2 quadrants. Each detected triomino is copied as its 2x2 bounding box into the quadrant whose corner shape it represents: missing lower-right goes to the upper-left quadrant; missing lower-left goes to upper-right; missing upper-right goes to lower-left; missing upper-left goes to lower-right. Colors are preserved. The objects may appear anywhere in the input and in any order.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    out = [[0 for _ in range(4)] for _ in range(4)]

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or seen[r][c]:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))

            r0 = min(rr for rr, _ in cells)
            c0 = min(cc for _, cc in cells)
            present = {(rr - r0, cc - c0) for rr, cc in cells}
            block = [[color if (dr, dc) in present else 0 for dc in range(2)] for dr in range(2)]

            if (1, 1) not in present:
                orow, ocol = 0, 0
            elif (1, 0) not in present:
                orow, ocol = 0, 2
            elif (0, 1) not in present:
                orow, ocol = 2, 0
            else:  # missing upper-left
                orow, ocol = 2, 2

            for dr in range(2):
                for dc in range(2):
                    out[orow + dr][ocol + dc] = block[dr][dc]
    return out
```

## Generator Constraints

ARC-GEN creates a 13x13 input and a 4x4 output. Four distinct nonzero colors are sampled with `common.random_colors(4)`. Four 2x2 bounding boxes are placed with upper-left rows and columns in `0..11`, and `common.overlaps(..., margin=1)` rejects placements that touch or overlap, so each triomino is isolated by at least one black cell from the others. There is exactly one object of each missing-corner orientation. Every object has area 3 and a 2x2 bounding box. No tie-breaking is needed because the four orientations are unique and colors are arbitrary.

## Reference Notes

The ARC-DSL solver finds all foreground objects, extracts the object whose selected bounding-box corner is black, takes its 2x2 subgrid, and concatenates the four subgrids into the final 4x4 output. The order is: object with missing lower-right, missing lower-left, missing upper-right, missing upper-left. The Code Golf solution encodes the same orientation-to-quadrant rule with compact indexing over flattened grid positions.
