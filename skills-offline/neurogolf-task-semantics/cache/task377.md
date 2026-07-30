# task377 Semantics

## Sources

- Current champion builder: `solutions_py/task377.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task377.json`
- ARC-GEN task id: `eb5a1d5d`
- ARC-DSL task id: `eb5a1d5d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_eb5a1d5d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_eb5a1d5d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task377.py`

## Pattern

The input is a large grid of nested filled rectangles. The first rectangle is the whole grid. Each next rectangle is strictly inside the previous one, has a different color from its immediate parent, and overwrites the cells in its bounding box. Colors may repeat at non-adjacent depths.

The output is a compact concentric-ring square. If the input has `n` nested rectangle colors in outside-to-inside order, the output size is `(2*n - 1) x (2*n - 1)`. Ring `0` uses the outermost color, ring `1` the next color, and so on until the center cell uses the innermost color.

## Readable Python Solver

```python
def solve(grid):
    colors = []
    sub = [row[:] for row in grid]

    while sub and sub[0]:
        color = sub[0][0]
        colors.append(color)
        pts = [
            (r, c)
            for r, row in enumerate(sub)
            for c, value in enumerate(row)
            if value != color
        ]
        if not pts:
            break
        r0, r1 = min(r for r, _ in pts), max(r for r, _ in pts)
        c0, c1 = min(c for _, c in pts), max(c for _, c in pts)
        sub = [row[c0:c1 + 1] for row in sub[r0:r1 + 1]]

    size = 2 * len(colors) - 1
    out = [[0 for _ in range(size)] for _ in range(size)]
    for i, color in enumerate(colors):
        for r in range(i, size - i):
            for c in range(i, size - i):
                out[r][c] = color
    return out
```

## Generator Constraints

- Input width and height are independently sampled from 20 through 30.
- The first rectangle covers the full input grid.
- Each child rectangle has width and height at least 3 during random generation and is placed at least one cell inside its parent. Validation examples include depths up to five and can end with width or height 4.
- The generator stops when a rectangle becomes too small or randomly after adding a child.
- Each child color is different from its immediate parent, but it may repeat an earlier outer color.
- Output size is determined only by nesting depth, not by the source rectangle dimensions or offsets.

## Reference Notes

ARC-GEN directly records the outside-to-inside colors and emits square rings of size `2*n-1`. ARC-DSL solves the same problem by deduplicating repeated adjacent rows/columns, mirroring the deduped color sequence, and forming the compact concentric square. The Code Golf solution is a very small recursive dedupe/transpose implementation, confirming that row/column compression is enough to recover the depth sequence.

There is no disagreement between references. The main edge case is repeated colors at non-adjacent depths; solvers must recover depth from geometry, not from the set of unique colors.
