# task097 Semantics

## Sources

- Current champion builder: `solutions_py/task097.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task097.json`
- ARC-GEN task id: `42a50994`
- ARC-DSL task id: `42a50994`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_42a50994.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_42a50994.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task097.py`

## Pattern

The input and output have the same rectangular shape, with dimensions from 5 to 20 in each axis. The background is black (`0`). A single nonzero foreground color is chosen for the whole example. Foreground pixels are scattered randomly. The output removes exactly the isolated one-cell foreground connected components under 8-neighborhood connectivity: a foreground cell is preserved if at least one of its eight neighbors is also foreground, and is changed to black if it has no foreground neighbor. Background cells remain black.

Equivalently, for every cell, count foreground cells in the clipped 3x3 window centered on that cell, including the center. A foreground center is kept when this count is greater than 1; otherwise the cell is black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 0:
                out[r][c] = 0
                continue
            friends = 0
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] != 0:
                        friends += 1
            out[r][c] = color if friends > 1 else 0
    return out
```

## Generator Constraints

The generator chooses width and height independently in `[5, 20]`. It independently samples candidate foreground cells with probability about 1/10, then paints every selected cell with the same randomly selected nonzero color. The output starts as a copy of the input and then blackens cells whose clipped 3x3 foreground count is exactly 1. There may be no selected cells, isolated border/corner cells, diagonal-only connected pairs, dense clusters, and multiple disconnected components. Out-of-bounds neighbors count as black/empty, not as background-channel cells.

## Reference Notes

The ARC-DSL solver computes connected objects, filters objects of size one, merges those singleton cells, and covers them with background. That matches the generator's 8-neighborhood rule because ARC objects are connected components using all eight directions. The Code Golf 2025 solution is the same idea in compact form: inspect each 3x3 neighborhood and zero cells that have no same-color/foreground companion.
