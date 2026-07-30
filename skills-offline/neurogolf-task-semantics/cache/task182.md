# task182 Semantics

## Sources

- Current champion builder: `solutions_py/task182.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task182.json`
- ARC-GEN task id: `776ffc46`
- ARC-DSL task id: `776ffc46`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_776ffc46.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_776ffc46.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task182.py`

## Pattern

The grid is normally `20x20`. It contains several small sprite objects, usually in colors blue (`1`) plus one prototype sprite in color red (`2`) or green (`3`). A gray (`5`) hollow rectangular box surrounds the prototype sprite. The output keeps the whole input, including gray boxes and all sprites, but recolors every sprite whose normalized shape matches the boxed prototype to the prototype color. Nonmatching sprites keep their original color. Fake/partial gray boxes and other colored nonmatching sprites may appear in examples, so the boxed prototype shape is the source of truth, not color alone.

## Readable Python Solver

```python
def solve(grid):
    H, W = len(grid), len(grid[0])

    def neigh4(r, c):
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                yield rr, cc

    def components(cells):
        cells = set(cells)
        out = []
        while cells:
            start = cells.pop()
            stack = [start]
            comp = {start}
            while stack:
                r, c = stack.pop()
                for nb in neigh4(r, c):
                    if nb in cells:
                        cells.remove(nb)
                        comp.add(nb)
                        stack.append(nb)
            out.append(comp)
        return out

    def bbox(cells):
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        return min(rs), min(cs), max(rs), max(cs)

    def border_cells(r0, c0, r1, c1):
        cells = set()
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        return cells

    gray = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 5]
    gray_boxes = []
    for comp in components(gray):
        r0, c0, r1, c1 = bbox(comp)
        if comp == border_cells(r0, c0, r1, c1):
            gray_boxes.append((r0, c0, r1, c1))

    # The intended box contains the colored prototype sprite.
    target = None
    for r0, c0, r1, c1 in gray_boxes:
        inside = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if grid[r][c] not in (0, 5):
                    inside.append((r, c, grid[r][c]))
        if inside:
            color = inside[0][2]
            cells = {(r, c) for r, c, v in inside if v == color}
            target = (color, cells)
            break
    if target is None:
        return [row[:] for row in grid]

    target_color, target_cells = target
    tr0, tc0, _, _ = bbox(target_cells)
    target_shape = {(r - tr0, c - tc0) for r, c in target_cells}

    # Recolor every non-gray, non-background object with the same normalized mask.
    object_cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row)
                    if v not in (0, 5)]
    out = [row[:] for row in grid]
    for comp in components(object_cells):
        r0, c0, _, _ = bbox(comp)
        shape = {(r - r0, c - c0) for r, c in comp}
        if shape == target_shape:
            for r, c in comp:
                out[r][c] = target_color
    return out
```

## Generator Constraints

ARC-GEN uses square grids of size `20`. Random cases place `5` or `6` non-overlapping sprites, each drawn from a fixed library of ten small masks fitting in a `5x5` footprint. At least one non-prototype copy exists because `idxs[1] = idxs[0]`. The prototype color is `2` or `3`; random non-prototype sprites are blue (`1`). A `7x7` gray box is drawn around the prototype at `(row-1, col-1)`. Hand validation cases include extra boxes, partial off-grid boxes, and occasional colored nonmatching sprites, so robust logic must find the complete gray box and use the sprite inside it as the prototype.

## Reference Notes

ARC-DSL selects gray objects whose indices equal their bounding box border, takes the `inbox`, extracts the nonzero object inside, normalizes its shape, then filters all non-gray/nonzero objects whose normalized index set matches and fills them with the prototype color. The Code Golf solution is a regex-based version of the same idea: find the gray boxed region, capture the prototype pattern, and substitute its color into matching copies.
