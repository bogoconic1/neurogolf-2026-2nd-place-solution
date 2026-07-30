# task034 Semantics

## Sources

- Current champion builder: `solutions_py/task034.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task034.json`
- ARC-GEN task id: `1f0c79e5`
- ARC-DSL task id: `1f0c79e5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1f0c79e5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1f0c79e5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task034.py`

## Pattern

The input is a `9 x 9` grid. Four colored cells form the corners of an axis-aligned `2 x 2` seed square. The object color is any non-red nonzero color. One to three of the four corner cells are red markers instead of object color. The output replaces the red markers with the object color and extends the object color outward from each marked corner along that corner's diagonal direction. Each extension step paints both the diagonal cell and the two orthogonally adjacent cells on the same diagonal front, stopping at the grid boundary.

For corner directions ordered northwest, northeast, southeast, southwest, a red marker means sprout in that direction from the corresponding seed corner. Unmarked seed corners remain as ordinary object-color cells.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    red = 2
    colors = sorted({v for row in grid for v in row if v not in (0, red)})
    color = colors[0]

    cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v in (red, color)]
    r0 = min(r for r, c in cells)
    c0 = min(c for r, c in cells)
    corners = [
        (r0, c0, -1, -1),
        (r0, c0 + 1, -1, 1),
        (r0 + 1, c0 + 1, 1, 1),
        (r0 + 1, c0, 1, -1),
    ]

    out = [row[:] for row in grid]
    for r, c, dr, dc in corners:
        if grid[r][c] != red:
            out[r][c] = color
            continue
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w:
            out[rr][cc] = color
            if 0 <= rr + dr < h:
                out[rr + dr][cc] = color
            if 0 <= cc + dc < w:
                out[rr][cc + dc] = color
            rr += dr
            cc += dc
    return out
```

## Generator Constraints

ARC-GEN fixes size `9`. The top-left corner of the `2 x 2` seed square is sampled with `row` and `col` in `1..6`, so all four seed corners are initially inside the grid with at least one cell of margin on the northwest side and at least one cell of margin on the southeast side. The sprout direction set contains one to three distinct directions from the four diagonal corners. The object color excludes red (`2`). The input contains only background `0`, red markers, and the object color. The output may touch any boundary depending on seed position and selected directions.

## Reference Notes

The ARC-DSL solver removes red, finds the least remaining nonzero color as the object color, combines the red cells with object-color seed cells to recover the full `2 x 2` seed object, normalizes red marker offsets against the seed's upper-left corner, maps those offsets to diagonal directions with `2*x - 1`, repeatedly shifts the recolored seed object along each marked diagonal, and paints all shifted copies onto the input. The Code Golf solution is a compact regex/rotation implementation of the same diagonal sprout expansion.
