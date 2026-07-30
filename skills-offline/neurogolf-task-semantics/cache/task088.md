# task088 Semantics

## Sources

- Current champion builder: `solutions_py/task088.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task088.json`
- ARC-GEN task id: `3de23699`
- ARC-DSL task id: `3de23699`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3de23699.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3de23699.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task088.py`

## Pattern

The input contains a rectangular box indicated only by its four corner marker pixels, plus a sparse sprite drawn inside the rectangle in a different color. The four marker pixels all share the output color. The sprite pixels all share a second color.

The output is the rectangle interior cropped to the box dimensions. Every sprite pixel inside the box is preserved at its relative position, but recolored from the sprite color to the marker color. All non-sprite cells in the cropped output are black. The corner marker pixels and the surrounding padding are not included in the output crop.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    colors = sorted({v for row in grid for v in row if v})
    if len(colors) != 2:
        return []
    cells_by_color = {
        c: [(r, col) for r, row in enumerate(grid) for col, v in enumerate(row) if v == c]
        for c in colors
    }
    marker = next(c for c, cells in cells_by_color.items() if len(cells) == 4)
    sprite = next(c for c in colors if c != marker)
    marker_cells = cells_by_color[marker]
    r0 = min(r for r, _ in marker_cells) + 1
    r1 = max(r for r, _ in marker_cells)
    c0 = min(c for _, c in marker_cells) + 1
    c1 = max(c for _, c in marker_cells)
    out = [[0 for _ in range(c1 - c0)] for _ in range(r1 - r0)]
    for r, c in cells_by_color[sprite]:
        if r0 <= r < r1 and c0 <= c < c1:
            out[r - r0][c - c0] = marker
    return out
```

## Generator Constraints

The box interior width `wide` and height `tall` are each sampled from 3..10. The full input width is sampled from `4 + wide` through `4 + 2 * wide`, and height from `4 + tall` through `4 + 2 * tall`. The top-left interior origin `(brow, bcol)` leaves a two-cell margin around the marker rectangle. Four marker pixels are placed at `(brow-1,bcol-1)`, `(brow-1,bcol+wide)`, `(brow+tall,bcol-1)`, and `(brow+tall,bcol+wide)`. The sprite has `wide + tall - 1`, `wide + tall`, or `wide + tall + 1` sampled cells inside the interior. The two colors are random non-background colors.

## Reference Notes

ARC-DSL partitions foreground objects, identifies the size-4 marker object, crops the subgrid spanned by the markers, trims the marker border away, and replaces the sprite color with the marker color. The Code Golf solution is a compact rotation/trim expression implementing the same marker-bounded crop.
