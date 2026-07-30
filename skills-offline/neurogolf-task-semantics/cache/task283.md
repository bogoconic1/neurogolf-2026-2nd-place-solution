# task283 Semantics

## Sources

- Current champion builder: `solutions_py/task283.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task283.json`
- ARC-GEN task id: `b6afb2da`
- ARC-DSL task id: `b6afb2da`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b6afb2da.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b6afb2da.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task283.py`

## Pattern

The input is a 10x10 grid containing two filled gray rectangles (`5`) on an otherwise black background. The rectangles are axis-aligned, may be transposed as a whole, have widths whose sum is 8 or 9, and heights from 3 through 6. The output recolors each gray rectangle by local position: interior cells become red (`2`), non-corner border cells become yellow (`4`), and rectangle corners become blue (`1`). Background stays black (`0`).

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 5:
                out[r][c] = 0
                continue
            up = r == 0 or grid[r - 1][c] != 5
            down = r == h - 1 or grid[r + 1][c] != 5
            left = c == 0 or grid[r][c - 1] != 5
            right = c == w - 1 or grid[r][c + 1] != 5
            vertical_edge = up or down
            horizontal_edge = left or right
            if vertical_edge and horizontal_edge:
                out[r][c] = 1
            elif vertical_edge or horizontal_edge:
                out[r][c] = 4
            else:
                out[r][c] = 2
    return out
```

## Generator Constraints

The generated grid size is fixed at 10x10 before NeuroGolf padding. There are exactly two gray rectangles. Each width is 3 through 6 and the two widths sum to 8 or 9; each height is 3 through 6. Rectangle rows are sampled independently so vertical positions may differ. Columns are chosen so one rectangle is left-ish and one right-ish with a possible gap/offset based on total width. The whole example may be transposed, so the same local border/corner rule must handle horizontal and vertical layouts. Only colors `0`, `1`, `2`, `4`, and `5` are semantically relevant.

## Reference Notes

ARC-DSL extracts gray objects, replaces all gray with red, fills each object box with yellow, then fills all object corners blue. The Code Golf solution is an obfuscated local-neighborhood recurrence, consistent with a 3x3 stencil: the output class for a gray cell depends only on whether its four axial neighbors are gray.
