# task033 Semantics

## Sources

- Current champion builder: `solutions_py/task033.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task033.json`
- ARC-GEN task id: `1e32b0e9`
- ARC-DSL task id: `1e32b0e9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1e32b0e9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1e32b0e9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task033.py`

## Pattern

The input is a square `3 x 3` board of large cells separated by full horizontal and vertical grid lines. The separating line color is the repeated nonzero frame color. Inside the top-left large cell there is a small colored template shape occupying positions in a `3 x 3` mini-pattern area offset one cell down/right from that large cell's corner. Other large cells may contain some pixels from the same template shape in a different object color.

The output keeps the input unchanged except for zero cells at template positions. It copies the top-left template's geometry into every large cell, using the frame/line color as the fill color. Existing object-colored pixels are preserved, so when a destination cell already contains a colored pixel at a template location, that pixel remains the object color rather than being overwritten by the frame color.

Equivalently: extract the nonzero non-frame object in the top-left cell, tile its relative coordinates to all nine large cells, and underfill those coordinates with the frame color.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    cell = (h - 2) // 3

    # The frame color is the most frequent nonzero color. In the generated tasks
    # it appears on the full grid lines; zeros are the background.
    counts = {}
    for row in grid:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0) + 1
    line = max(counts, key=counts.get)

    # Template pixels are the nonzero, non-frame pixels in the top-left cell.
    template = []
    for r in range(cell):
        for c in range(cell):
            v = grid[r][c]
            if v != 0 and v != line:
                template.append((r, c))

    out = [row[:] for row in grid]
    for mr in range(3):
        for mc in range(3):
            ro = mr * (cell + 1)
            co = mc * (cell + 1)
            for r, c in template:
                rr = ro + r
                cc = co + c
                if out[rr][cc] == 0:
                    out[rr][cc] = line
    return out
```

## Generator Constraints

ARC-GEN builds `common.hollywood_squares(3, 0, linecolor, size)`, whose side length is `3 * size + 2`. The default and generated size is `5`, so the standard grids are `17 x 17`, with grid lines at row/column `5` and `11`. The template coordinates are chosen from three groups within a `3 x 3` mini-pattern: corners, edge centers, and center. Two or three of these groups are selected, so the top-left template has between five and nine pixels. The template object color and line color are distinct random nonzero colors. Random extra copies are sampled only in the lower-right `2 x 2` large cells, while fixed validation examples also cover copies in top-row and left-column cells. Existing colored pixels must therefore be preserved in every cell and not overwritten by line-color template fills.

## Reference Notes

The ARC-DSL solver computes `cell = (height - 2) / 3`, crops the top-left cell, extracts the nonzero object there, computes the frame color as the remaining non-background color, shifts the top-left object to all nine cell origins, and underfills the original input with that frame color. The Code Golf solution is a compact recursive version of the same rule: derive the top-left template and tile it through the board while keeping existing nonzero pixels.
