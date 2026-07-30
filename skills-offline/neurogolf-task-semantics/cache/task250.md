# task250 Semantics

## Sources

- Current champion builder: `solutions_py/task250.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task250.json`
- ARC-GEN task id: `a48eeaf7`
- ARC-DSL task id: `a48eeaf7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a48eeaf7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a48eeaf7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task250.py`

## Pattern

The input is a square grid containing a solid 2x2 red block (`2`) and zero or more gray pixels (`5`) somewhere around it. The output keeps the red block fixed, removes all original gray pixels, and redraws each gray pixel at the closest cell on the one-cell-thick rectangular border surrounding the red 2x2 block. Equivalently, if the red block has top-left corner `(br, bc)`, each gray pixel `(r, c)` is moved to `(clamp(r, br-1, br+2), clamp(c, bc-1, bc+2))`. Background remains black (`0`).

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    red = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 2]
    br = min(r for r, _ in red)
    bc = min(c for _, c in red)

    out = [[0 for _ in range(w)] for _ in range(h)]
    for dr in (0, 1):
        for dc in (0, 1):
            out[br + dr][bc + dc] = 2

    lo_r, hi_r = br - 1, br + 2
    lo_c, hi_c = bc - 1, bc + 2
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5:
                rr = min(max(r, lo_r), hi_r)
                cc = min(max(c, lo_c), hi_c)
                out[rr][cc] = 5
    return out
```

## Generator Constraints

ARC-GEN uses 10x10 grids by default. The red 2x2 block top-left corner is sampled with row and column in `[2, 6]`, so the surrounding 4x4 border always lies inside the grid. Gray pixels are generated from the 12 sectors/cells around the red block; each sector may be skipped independently, and when present the pixel is placed some positive distance outward from that border cell while respecting grid bounds. There can be no gray pixel inside the red 2x2 block. The output is always the same size as the input.

## Reference Notes

The ARC-DSL solver computes `outbox` of the red object, then for every gray pixel chooses the outbox cell with minimum Manhattan distance and fills those positions after covering/removing the original gray pixels. The generator shows this nearest-cell choice is exactly row/column clamping to the 4x4 border around the 2x2 red block. The Code Golf 2025 solution implements the same idea by repeatedly transposing/reversing the grid and projecting gray occupancy toward the red-box boundary.
