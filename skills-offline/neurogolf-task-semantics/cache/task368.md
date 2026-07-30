# task368 Semantics

## Sources

- Current champion builder: `solutions_py/task368.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task368.json`
- ARC-GEN task id: `e76a88a6`
- ARC-DSL task id: `e76a88a6`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e76a88a6.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e76a88a6.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task368.py`

## Pattern

The input contains several same-size rectangular sprite slots on a 10x10 grid. One slot is the colored exemplar template, using two non-gray colors. The other slots are gray (`5`) placeholders with the same shape/extent. The output keeps the exemplar and fills every gray placeholder with the exemplar color pattern aligned to that placeholder upper-left corner. Grid size is unchanged.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 0 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(cells)
    exemplar = max(comps, key=lambda comp: len({grid[r][c] for r, c in comp}))
    er0 = min(r for r, _ in exemplar)
    ec0 = min(c for _, c in exemplar)
    template = {(r - er0, c - ec0): grid[r][c] for r, c in exemplar}
    for comp in comps:
        if comp is exemplar:
            continue
        r0 = min(r for r, _ in comp)
        c0 = min(c for _, c in comp)
        for (dr, dc), color in template.items():
            out[r0 + dr][c0 + dc] = color
    return out
```

## Generator Constraints

ARC-GEN uses a 10x10 grid. Template width and height are 3 or 4, with at least one dimension equal to 3. There are 3 or 4 non-overlapping sprite rectangles. The first sprite is colored by a fixed `width*height` list of two non-gray colors; every later sprite is gray in the input. In the output, every sprite rectangle receives the exemplar colors. Sprite rectangles are separated enough to avoid overlap.

## Reference Notes

The ARC-DSL solver finds all non-background objects, selects the object with the most distinct colors as the template, normalizes it to relative coordinates, then shifts that normalized template to the upper-left corners of the other objects and paints it. The Code Golf 2025 solution encodes a similar scan/template-fill behavior compactly over rows.
