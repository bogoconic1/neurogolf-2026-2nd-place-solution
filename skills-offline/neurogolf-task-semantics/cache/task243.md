# task243 Semantics

## Sources

- Current champion builder: `solutions_py/task243.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task243.json`
- ARC-GEN task id: `9edfc990`
- ARC-DSL task id: `9edfc990`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_9edfc990.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_9edfc990.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task243.py`

## Pattern

The input is a square grid containing arbitrary nonzero colored cells, black (`0`) cells, and one or more blue (`1`) seed cells. Flood-fill from every blue seed through 4-connected black cells only. Every black cell in a connected component adjacent to the blue region becomes blue; all nonzero non-blue colors remain barriers and are preserved. The output size equals the input size.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    q = []
    for r in range(h):
        for c in range(w):
            if out[r][c] == 1:
                q.append((r, c))

    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and out[rr][cc] == 0:
                out[rr][cc] = 1
                q.append((rr, cc))
    return out
```

## Generator Constraints

ARC-GEN chooses square size `12..18`. Each cell is independently black with probability about one half, otherwise a random color is chosen; cells equal to blue (`1`) become the initial flood seeds. The output is produced by repeatedly popping blue cells and painting neighboring black cells blue, so the fill is exactly 4-connected and cannot cross any nonzero non-blue color. The generated grids are embedded in NeuroGolf's dense `[1,10,30,30]` tensor contract, but the active semantic crop is at most 18x18.

## Reference Notes

The ARC-DSL solver finds zero-colored connected objects, selects those adjacent to any color-1 cell, recolors the selected zero components to color 1, and paints them back into the input. This confirms the component interpretation: the fill reaches whole black components adjacent to blue, while other colors are barriers. The Code Golf solution repeatedly rotates/transposes the grid and replaces adjacent `1,0` pairs with `1,1`, which is an iterative flood-fill implementation.
