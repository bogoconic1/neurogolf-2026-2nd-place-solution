# task157 Semantics

## Sources

- Current champion builder: `solutions_py/task157.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task157.json`
- ARC-GEN task id: `6a1e5592`
- ARC-DSL task id: `6a1e5592`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6a1e5592.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6a1e5592.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task157.py`

## Pattern

The input is a fixed `10x15` grid. Rows `0..2` start as a red (`2`) bar, but black (`0`) cutouts inside that red band show the top footprint of several hidden shapes. Matching gray (`5`) shapes appear lower in the grid, bottom-aligned in rows `6..9`. The output removes every gray shape from the lower area, keeps the red bar and background otherwise, and paints the matched full shapes blue (`1`) at their corresponding top cutout locations.

Each gray creature must be matched to one black cutout in the red top band by footprint. If the cutout begins on row 1, only the top rows visible above it are available, so matching uses the visible footprint against the top one or two rows of each gray creature as appropriate. Generator constraints make these visible footprints unique, so greedy exact matching is unambiguous.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 if value == 5 else value for value in row] for row in grid]

    # Extract each lower gray creature as a normalized 4-connected shape.
    seen = set()
    shapes = []
    for r in range(3, h):
        for c in range(w):
            if grid[r][c] != 5 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            component = []
            while stack:
                y, x = stack.pop()
                component.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 3 <= yy < h and 0 <= xx < w and grid[yy][xx] == 5 and (yy, xx) not in seen:
                        seen.add((yy, xx))
                        stack.append((yy, xx))
            r0 = min(y for y, _ in component)
            c0 = min(x for _, x in component)
            cells = {(y - r0, x - c0) for y, x in component}
            width = max(x for _, x in cells) + 1
            shapes.append((cells, width))

    # Every black cell in the red top band belongs to exactly one visible
    # footprint. Match all creatures jointly because a footprint can be
    # disconnected and a one-row footprint can be locally ambiguous.
    black = {(r, c) for r in range(3) for c in range(w) if grid[r][c] == 0}
    options = []
    for cells, width in shapes:
        placements = []
        for top in (1, 2):
            for left in range(w - width + 1):
                visible = {(top + r, left + c) for r, c in cells if top + r < 3}
                if visible and visible <= black:
                    placements.append((top, left, visible))
        options.append(placements)

    order = sorted(range(len(shapes)), key=lambda i: len(options[i]))
    chosen = [None] * len(shapes)

    def match(k, covered):
        if k == len(order):
            return covered == black
        i = order[k]
        for top, left, visible in options[i]:
            if covered.isdisjoint(visible):
                chosen[i] = (top, left)
                if match(k + 1, covered | visible):
                    return True
        chosen[i] = None
        return False

    if not match(0, set()):
        raise ValueError("no exact footprint matching")

    for (cells, _), (top, left) in zip(shapes, chosen):
        for r, c in cells:
            out[top + r][left + c] = 1
    return out
```

## Generator Constraints

The generator uses `width=15`, `height=10`, and `num_shapes=max(3, randint(0,4))`, so there are 3 or 4 shapes. Each shape has width `1..4`, height `2..4`, and a continuous connected creature occupying between roughly half and all cells in its bounding box. Blue/top locations are placed with top row `1` or `2`; gray/lower copies are bottom-aligned by shape height (`grayrow = height - max_row - 1`). The blue/top boxes and gray/lower boxes are placed without overlap using a one-cell margin. The red top band must remain vertically convex after black cutouts are drawn, and the generator rejects examples where any visible top footprint could match another creature's one-row or two-row footprint. There is no rotation, reflection, recoloring choice, or variable canvas size.

## Reference Notes

ARC-DSL extracts gray objects, removes them from the input, normalizes each gray shape, then scores all possible shifts inside the top `5 x width` area against red, black, and shape-fit features. It fills the best-matching shifted gray-shape indices with blue. The Code Golf solution is heavily compressed but implements the same footprint matching over the flattened `10x15` grid. The references agree that gray shapes are not copied by position; they are matched by their top visible footprints and stamped into the black cutouts.
