# task131 Semantics

## Sources

- Current champion builder: `solutions_py/task131.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task131.json`
- ARC-GEN task id: `56dc2b01`
- ARC-DSL task id: `56dc2b01`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_56dc2b01.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_56dc2b01.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task131.py`

## Pattern

The input contains a green connected shape and a red line. In the untransposed canonical orientation, the grid is height `4..5` and width `16..18`; the green shape is near the left side and the red line is a full vertical line near the right side. The output moves the green shape horizontally until its right edge sits immediately next to the red line. A cyan guide line is drawn one cell beyond the moved shape on the side away from the red line. The red line remains red. The generator may horizontally flip and/or transpose the whole input and output, so the same rule must handle horizontal or vertical motion and either side of the red line.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    greens = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 3]
    reds = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 2]

    green_rows = [r for r, _ in greens]
    green_cols = [c for _, c in greens]
    red_rows = [r for r, _ in reds]
    red_cols = [c for _, c in reds]

    vertical_red = len(set(red_cols)) == 1
    if vertical_red:
        red_c = red_cols[0]
        if max(green_cols) < red_c:
            shift = red_c - max(green_cols) - 1
            cyan_c = min(c for _, c in greens) + shift - 1
        else:
            shift = red_c - min(green_cols) + 1
            cyan_c = max(c for _, c in greens) + shift + 1
        for r, c in greens:
            out[r][c] = 0
        for r, c in greens:
            out[r][c + shift] = 3
        for r in range(h):
            out[r][cyan_c] = 8
        for r in range(h):
            out[r][red_c] = 2
    else:
        red_r = red_rows[0]
        if max(green_rows) < red_r:
            shift = red_r - max(green_rows) - 1
            cyan_r = min(r for r, _ in greens) + shift - 1
        else:
            shift = red_r - min(green_rows) + 1
            cyan_r = max(r for r, _ in greens) + shift + 1
        for r, c in greens:
            out[r][c] = 0
        for r, c in greens:
            out[r + shift][c] = 3
        for c in range(w):
            out[cyan_r][c] = 8
        for c in range(w):
            out[red_r][c] = 2
    return out
```

## Generator Constraints

ARC-GEN samples width `16..18`, height `4..5`, a continuous green shape with `8..9` cells spanning the height, a small left offset `0..3`, and a red line coordinate near the far side. The red line is drawn through every row in canonical orientation. The generated pair is optionally flipped horizontally and optionally transposed. Colors are fixed: green `3`, red `2`, cyan `8`, black `0`.

## Reference Notes

The ARC-DSL solver finds the green object and red cells, uses `gravitate` to compute the shift needed to move the green object against the red line, paints a shifted cyan line, then moves the green object. The Code Golf solution compactly sorts/transposes rows and recurses to implement the same move-toward-red-line behavior under orientation changes.
