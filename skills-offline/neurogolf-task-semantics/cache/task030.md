# task030 Semantics

## Sources

- Current champion builder: `solutions_py/task030.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task030.json`
- ARC-GEN task id: `1caeab9d`
- ARC-DSL task id: `1caeab9d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1caeab9d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1caeab9d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task030.py`

## Pattern

The input contains three copies of the same small diagonally connected shape, colored 1, 2, and 4, on a black grid of width 10 and height 5 or 10. The copies appear in separate horizontal bands/columns, with possibly different top rows. The output erases the original copies and repaints each object in the same columns but vertically shifted so its bottom row aligns with the bottom row of the color-1 object. Color 1 therefore stays in place, while colors 2 and 4 move up or down to share color 1's vertical baseline.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]

    ones = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 1]
    target_bottom = max(r for r, _ in ones)

    for color in (1, 2, 4):
        cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == color]
        bottom = max(r for r, _ in cells)
        dr = target_bottom - bottom
        for r, c in cells:
            out[r + dr][c] = color
    return out
```

## Generator Constraints

ARC-GEN fixes `width=10` and chooses `height` as 5 or 10. The base mini-shape has width and height 2 or 3, contains between half and all of its cells, and is diagonally connected. The same shape is drawn once in each color `(1, 2, 4)`. Each copy gets an independent vertical placement that keeps the shape inside the grid. The three horizontal starts come from `[0, miniwidth, 2*miniwidth]`, then a suffix may be shifted one cell right and the starts are shuffled, so columns are data-dependent but remain within width 10. Output uses the color-1 copy's vertical placement for all three colors.

## Reference Notes

ARC-DSL finds all objects, computes the lowermost row of color 1, then shifts every object by the vertical vector from its own lowermost row to the color-1 lowermost row. The merged original objects are covered before the shifted objects are painted.
