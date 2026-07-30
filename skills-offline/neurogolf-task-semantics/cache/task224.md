# task224 Semantics

## Sources

- Current champion builder: `solutions_py/task224.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task224.json`
- ARC-GEN task id: `928ad970`
- ARC-DSL task id: `928ad970`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_928ad970.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_928ad970.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task224.py`

## Pattern

The input contains four gray (`5`) guide pixels around a hidden outer rectangle plus a visible smaller rectangle drawn in one non-gray color. The top/bottom guide pixels lie just outside the outer rectangle, and the left/right guide pixels lie just outside its sides. The visible inner rectangle supplies the foreground color.

The output keeps the original grid unchanged and additionally draws only the one-cell border just inside the gray guide bounds. If the first gray row is `top`, last gray row is `bottom`, first gray column is `left`, and last gray column is `right`, paint cells satisfying:

```text
top < r < bottom
left < c < right
and (r == top + 1 or r == bottom - 1 or c == left + 1 or c == right - 1)
```

using the non-gray/non-background foreground color already present in the input. The gray guide pixels remain gray because they are outside the painted border. The output size is the same grid size as the input, embedded in the standard NeuroGolf `[1, 10, 30, 30]` tensor.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    gray = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 5]
    top = min(r for r, _ in gray)
    bottom = max(r for r, _ in gray)
    left = min(c for _, c in gray)
    right = max(c for _, c in gray)

    colors = {v for row in grid for v in row if v not in (0, 5)}
    color = min(colors)

    out = [row[:] for row in grid]
    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if r in (top + 1, bottom - 1) or c in (left + 1, right - 1):
                out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN creates a black grid with variable `height` and `width`. The hidden outer rectangle has width and height in `8..11`, then the full grid adds `3..5` cells in each dimension. The outer rectangle starts away from the grid edge. A smaller visible rectangle has width and height in `3..4` and is placed strictly inside the outer rectangle with at least one-cell separation from the outer border.

There are exactly four gray guide pixels: one above the top side, one below the bottom side, one left of the rectangle, and one right of it. The top/bottom guide columns are inside the outer rectangle span. The left/right guide rows are inside the outer rectangle span. The foreground rectangle color is chosen from `1..9` excluding gray `5`.

The generator output draws both the hidden outer rectangle border and the visible inner rectangle in the chosen color, while preserving the four gray guide pixels. The input draws only the gray guide pixels and the inner rectangle.

## Reference Notes

The ARC-DSL solver computes all gray positions, takes their bounding subgrid, trims that subgrid, finds the least color in the trimmed region, and fills the `inbox` of the gray guide set. Because the only nonzero color inside the trimmed guide box is the visible inner rectangle color, `leastcolor` recovers the paint color. `inbox` corresponds to the one-cell border just inside the guide bounds, not the whole interior.

The Code Golf 2025 solution repeatedly rotates/transposes the grid while using the gray guide pixels to decide when to paint rows/columns. It confirms the same dynamic guide-frame rule but is highly compressed and not suitable as an ONNX structure by itself.
