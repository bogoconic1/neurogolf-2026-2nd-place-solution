# task145 Semantics

## Sources

- Current champion builder: `solutions_py/task145.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task145.json`
- ARC-GEN task id: `6455b5f5`
- ARC-DSL task id: `6455b5f5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6455b5f5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6455b5f5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task145.py`

## Pattern

The input is a top-left valid grid of size 10..20 by 10..20 containing only black background cells and red cut lines. The red lines come from recursively bisecting the full rectangle with complete horizontal or vertical divider lines, so they partition the black background into rectangular leaf regions. The output keeps every red line unchanged, colors every cell of each smallest-area black leaf region cyan (`8`), colors every cell of the largest-area leaf region blue (`1`), and leaves the other leaf regions black (`0`). If several zero regions share the minimum area, all of them become cyan. The generator rejects cases where the minimum and maximum leaf area are equal, so blue and cyan targets do not collapse to the same full set of regions.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    seen = [[False for _ in range(w)] for _ in range(h)]
    regions = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] == 2:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        if not seen[nr][nc] and grid[nr][nc] != 2:
                            seen[nr][nc] = True
                            stack.append((nr, nc))
            regions.append(cells)

    sizes = [len(cells) for cells in regions]
    min_size = min(sizes)
    max_size = max(sizes)
    out = [row[:] for row in grid]
    for cells, size in zip(regions, sizes):
        color = 8 if size == min_size else 1 if size == max_size else 0
        for r, c in cells:
            out[r][c] = color
    return out
```

## Generator Constraints

The generated valid grid width and height are independently sampled from 10 through 20. Starting from the whole rectangle, the generator recursively chooses no cut, a horizontal cut, or a vertical cut. Horizontal cuts are only allowed when the current region has height greater than 2; vertical cuts are only allowed when width is greater than 2. The cutline is strictly interior and is written as a full red divider, then the two sides are recursed independently. Leaf regions are therefore axis-aligned rectangles separated by one-cell red divider rows or columns. The generator resamples until at least two different leaf areas exist. The submitted NeuroGolf model still emits the standard dense `[1, 10, 30, 30]` one-hot output; cells outside the valid input rectangle are background in the padded representation.

## Reference Notes

The ARC-DSL solver uses `objects(I, T, F, F)`, then selects all zero-colored objects with minimum size for cyan, and the single largest object among all objects for blue. Because red cut lines are connected through intersections, the red object can be large, but in the generator examples the largest selected object is the largest partition region that becomes blue. The Code Golf solution rotates the grid repeatedly and propagates packed area-derived values; it supports the same view that the task is an extremal connected-component area fill over red-separated cells, not a simple local recolor.
