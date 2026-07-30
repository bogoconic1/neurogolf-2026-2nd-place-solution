# task225 Semantics

## Sources

- Current champion builder: `solutions_py/task225.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task225.json`
- ARC-GEN task id: `93b581b8`
- ARC-DSL task id: `93b581b8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_93b581b8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_93b581b8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task225.py`

## Pattern

The input is a 6x6 black canvas containing one nonzero 2x2 color block. The output preserves that source block and copies each of its four colors into a 2x2 stamp at the opposite diagonal/corner side: top-left source color stamps bottom-right, top-right stamps bottom-left, bottom-left stamps top-right, and bottom-right stamps top-left. Cells on horizontal/vertical frontiers through the original 2x2 merged object are cleared back to black, which prevents overpainting the cross lines between stamps.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    cells = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != 0]
    r0 = min(r for r, c, v in cells)
    c0 = min(c for r, c, v in cells)
    colors = {
        "tl": grid[r0][c0],
        "tr": grid[r0][c0 + 1],
        "bl": grid[r0 + 1][c0],
        "br": grid[r0 + 1][c0 + 1],
    }
    for name, rr, cc in [
        ("tl", r0 + 2, c0 + 2),
        ("tr", r0 + 2, c0 - 2),
        ("bl", r0 - 2, c0 + 2),
        ("br", r0 - 2, c0 - 2),
    ]:
        color = colors[name]
        for dr in (0, 1):
            for dc in (0, 1):
                r, c = rr + dr, cc + dc
                if 0 <= r < h and 0 <= c < w:
                    out[r][c] = color
    # Clear the frontiers through the source object's reflected support except source cells.
    source = {(r, c) for r, c, v in cells}
    reflected = {(r0 + dr, c0 + dc) for dr in (0, 1) for dc in (0, 1)}
    for r, c in reflected:
        for cc in range(w):
            if (r, cc) not in source:
                out[r][cc] = 0
        for rr in range(h):
            if (rr, c) not in source:
                out[rr][c] = 0
    return out
```

## Generator Constraints

ARC-GEN uses a fixed 6x6 canvas. The 2x2 source block origin has row and column each in `1..3`, so all four opposite 2x2 stamps are within bounds. The four source colors are random nonzero colors and may repeat only if the random color helper allows it; the submitted graph treats colors independently by scalar labels. Output uses the same colors and black background.

## Reference Notes

The ARC-DSL solver partitions foreground, mirrors the merged 2x2 object across both diagonals, upscales by 3, shifts by `(-2,-2)`, underpaints it onto the input, then clears horizontal and vertical frontiers through the mirrored support outside the original object. The Code Golf solution encodes the same role-dependent 2x2 stamp movement with regex/transpose recursion.
