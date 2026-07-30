# task193 Semantics

## Sources

- Current champion builder: `solutions_py/task193.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task193.json`
- ARC-GEN task id: `7f4411dc`
- ARC-DSL task id: `7f4411dc`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7f4411dc.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7f4411dc.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task193.py`

## Pattern

The input is a square grid with black background and one nonzero foreground color. The foreground forms several filled axis-aligned rectangles plus extra single-pixel same-color static/noise cells. The output keeps only the filled rectangles and deletes every static/noise cell back to black.

Equivalently, for each foreground cell, count its four direct neighbors up/down/left/right that are also foreground. Keep the cell if it has at least two foreground direct neighbors; otherwise set it to black. Rectangle corners have exactly two direct foreground neighbors, non-corner rectangle cells have more, and the generator rejects static pixels that would connect shapes or touch a box in more than one direct-neighbor position.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                continue
            n = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == grid[r][c]:
                    n += 1
            if n < 2:
                out[r][c] = 0
    return out
```

## Generator Constraints

- Input and output are square, with size in `7..20`.
- There is exactly one nonzero foreground color.
- The number of boxes is `size // 3 - 1`.
- Each filled rectangle has width and height in `2..5`.
- Rectangles are non-overlapping with one-cell spacing.
- Extra same-color static pixels are sampled with probability `0.05`, then filtered to remove neighboring static pixels.
- A static pixel is drawn only if it does not directly bridge two colored regions: it cannot have both left and right foreground neighbors, and it cannot have both top and bottom foreground neighbors.
- The generator also rejects samples where any box touches more than one static foreground pixel directly on its border.
- The generator loops until the input differs from the output, so at least one removable static pixel exists.

## Reference Notes

- ARC-DSL finds the least/nonzero color, selects foreground cells whose direct neighborhood contains more than two non-foreground positions, and fills those selected cells with black. This is the same as deleting foreground cells with fewer than two direct foreground neighbors.
- The Code Golf 2025 solution is a compact two-pass local min/max expression. It implements the same local cleanup: preserve cells supported by adjacent foreground structure and remove isolated static cells.
- Because all nonzero cells share one color, no color selection or rectangle component labeling is needed.
