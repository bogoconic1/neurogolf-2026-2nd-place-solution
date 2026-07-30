# task277 Semantics

## Sources

- Current champion builder: `solutions_py/task277.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task277.json`
- ARC-GEN task id: `b230c067`
- ARC-DSL task id: `b230c067`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b230c067.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b230c067.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task277.py`

## Pattern

Input grids are 10x10 with cyan (`8`) objects. There are three placed sprites: two duplicate sprites with the same normalized shape and one odd sprite made by deleting one column from that same shape. The output recolors the two duplicate/wider sprites blue (`1`) and the odd/narrower sprite red (`2`), leaving background black (`0`).

ARC-DSL expresses this as: take all foreground objects, normalize each object shape, find the least-common normalized shape, extract the object matching it, replace all cyan with blue, then fill the least-common object red. In generated data, the least-common object is the column-deleted narrower sprite; the two common objects are the original wider sprite.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    objects = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] != 8:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 8:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            objects.append(cells)
    def norm(cells):
        r0 = min(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        return frozenset((r - r0, c - c0) for r, c in cells)
    counts = {}
    for obj in objects:
        counts[norm(obj)] = counts.get(norm(obj), 0) + 1
    odd_shape = min(counts, key=counts.get)
    out = [[1 if v == 8 else v for v in row] for row in grid]
    for obj in objects:
        if norm(obj) == odd_shape:
            for r, c in obj:
                out[r][c] = 2
    return out
```

## Generator Constraints

The grid size is fixed at `10`. A base sprite has width `3..4` and height `2..4`. It is either a rectangular perimeter or a connected creature that uses every base column. A second sprite is derived by deleting one randomly chosen column, so its width is one less. The generator places two copies of the base sprite and one copy of the narrowed sprite without overlap. All input sprite cells are cyan; output cells from the base copies are blue and cells from the narrowed copy are red.

## Reference Notes

The ARC-DSL `leastcommon(normalize(objects))` confirms the normalized shape frequency rule, not an absolute position or color rule. The Code Golf solution uses repeated local propagation to spread object/width information and recolor the minority shape. The generator structure implies: because the odd sprite is exactly one column narrower than the duplicates, width propagation is enough to classify objects.
