# task029 Semantics

## Sources

- Current champion builder: `solutions_py/task029.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task029.json`
- ARC-GEN task id: `1c786137`
- ARC-DSL task id: `1c786137`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1c786137.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1c786137.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task029.py`

## Pattern

The input is a random colored grid with one rectangular zoom frame drawn in a color that does not appear in the random static. The answer is the subgrid strictly inside that frame: remove the one-cell border and return the enclosed colors and black cells at their original scale. The output size is the frame interior height by interior width.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    colors = sorted({v for row in grid for v in row if v})

    best = None
    for color in colors:
        cells = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == color]
        if not cells:
            continue
        r0 = min(r for r, _ in cells)
        r1 = max(r for r, _ in cells)
        c0 = min(c for _, c in cells)
        c1 = max(c for _, c in cells)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        ok = True
        for r in range(r0, r1 + 1):
            ok &= grid[r][c0] == color and grid[r][c1] == color
        for c in range(c0, c1 + 1):
            ok &= grid[r0][c] == color and grid[r1][c] == color
        if not ok:
            continue

        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if best is None or area > best[0]:
            best = (area, r0, r1, c0, c1)

    _, r0, r1, c0, c1 = best
    return [row[c0 + 1:c1] for row in grid[r0 + 1:r1]]
```

## Generator Constraints

ARC-GEN chooses width and height independently in `10..25`. It chooses three or four non-border colors, fills each cell independently with either black or one of those colors, then chooses a `zoom_color` excluded from the static colors. The frame interior has `zoom_width in 1..width-2` and `zoom_height in 1..height-2`; its top-left interior coordinate is at least one cell from every outer grid edge. The frame is one cell thick and overwrites any static on its perimeter. The output is exactly the pre-border interior crop, so it may contain black and any of the static colors, but never the `zoom_color`.

## Reference Notes

ARC-DSL identifies objects, selects the object with maximum height, takes its subgrid, and trims the border. This matches the generator because the zoom frame is the distinctive large rectangular object. The Code Golf solution repeatedly filters/rotates by color to isolate the framed region and returns the crop after removing the border. The references agree that the border color itself is not copied to the output.
