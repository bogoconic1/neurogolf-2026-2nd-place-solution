# task272 Semantics

## Sources

- Current champion builder: `solutions_py/task272.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task272.json`
- ARC-GEN task id: `aedd82e4`
- ARC-DSL task id: `aedd82e4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_aedd82e4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_aedd82e4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task272.py`

## Pattern

The input is a black rectangular grid, width and height each in `3..5`, with red cells (`2`) at arbitrary generated positions. The output preserves all red cells except isolated single-cell red objects: any red object of size exactly one is recolored blue (`1`). Black cells remain black. In other words, a red cell becomes blue iff it has no 4-connected red neighbor; red cells in 4-connected components of size greater than one remain red.

The submitted NeuroGolf graph still has the standard dense `[1,10,30,30]` output, but the real task canvas is top-left aligned and at most `5x5`.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 2:
                continue
            neighbor_red = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == 2:
                    neighbor_red = True
                    break
            if not neighbor_red:
                out[r][c] = 1
    return out
```

## Generator Constraints

From ARC-GEN `task_aedd82e4.py`:

- `width` is 3..5.
- `height` is 3..5.
- The input and output grids are black by default.
- The generator chooses a random subset of pixels in the rectangular grid and colors them red.
- The output starts as a copy of the input, then every edge-free red pixel is recolored blue.
- Colors used are black `0`, blue `1`, and red `2`.
- There are no other colors or objects; all reasoning is local to red 4-connectivity and the variable grid extent.

The variable width/height matters: cells outside the true top-left rectangle must stay outside/zero in the dense NeuroGolf output and must not be treated as ordinary black background.

## Reference Notes

ARC-DSL finds color-2 objects, filters those with size one, merges them, and fills those positions with color one. This confirms that the rule is component-size-one recoloring, not a diagonal isolation or bounding-box rule.

The Code Golf solution implements the same connectedness test by repeated scans/propagation over the grid. Its compact expression is hard to read directly, but it agrees with ARC-GEN and ARC-DSL: isolated red cells are converted to blue while connected red cells stay red.
