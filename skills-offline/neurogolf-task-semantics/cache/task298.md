# task298 Semantics

## Sources

- Current champion builder: `solutions_py/task298.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task298.json`
- ARC-GEN task id: `bda2d7a6`
- ARC-DSL task id: `bda2d7a6`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_bda2d7a6.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_bda2d7a6.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task298.py`

## Pattern

The input is a 6x6 or 8x8 square that is symmetric across both axes. Cell color is determined by depth from the top-left corner within each mirrored quadrant: `min(local_row, local_col) % 3`. Three colors are used and may include black. The output preserves the geometry and replaces each depth color with the previous color in the 3-color cycle.

Equivalently, identify the three colors ordered by nested depth from the outside inward, then repaint every cell of depth class `d` with the color from depth class `(d - 1) mod 3`.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    half = n // 2
    colors = [grid[i][i] for i in range(3)]
    out = [row[:] for row in grid]
    for r in range(n):
        rr = min(r, n - 1 - r)
        for c in range(n):
            cc = min(c, n - 1 - c)
            depth = min(rr, cc) % 3
            out[r][c] = colors[(depth - 1) % 3]
    return out
```

## Generator Constraints

- `half` is sampled from `3..4`, so grid size is either 6x6 or 8x8.
- Exactly three colors are sampled from `0..9`; black can be one of the colors.
- For each quadrant-local coordinate `(r, c)` in the top-left half, the input color index is `min(r, c) % 3` and the output color index is `(min(r, c) + 2) % 3`.
- The same values are mirrored to all four quadrants, so the full grid is horizontally and vertically symmetric.
- The output size equals the input size; no cropping or translation occurs.

## Reference Notes

The ARC-DSL solver partitions objects by color, orders them by object size, rotates the last color to the front, and repaints each partition with the shifted color. The Code Golf solution indexes the observed row-3 color order to perform the same 3-cycle replacement compactly.
