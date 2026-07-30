# task270 Semantics

## Sources

- Current champion builder: `solutions_py/task270.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task270.json`
- ARC-GEN task id: `ae3edfdc`
- ARC-DSL task id: `ae3edfdc`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ae3edfdc.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ae3edfdc.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task270.py`

## Pattern

The input is a `15x15` grid with two flower centers and sparse petals on the same row or column as their center. One flower has center color `2` and petal color `3`; the other has center color `1` and petal color `7`. In the input, each petal is a single pixel somewhere farther away from its center along one of the four cardinal directions. The output keeps the two centers fixed, removes the distant petal pixels, and redraws each existing petal in the adjacent cell next to its own center in the same direction. Missing petals stay missing. Background is `0`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 if grid[r][c] in (3, 7) else grid[r][c] for c in range(w)] for r in range(h)]

    centers = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] in (1, 2):
                centers[grid[r][c]] = (r, c)

    for petal, center_color in ((3, 2), (7, 1)):
        cr, cc = centers[center_color]
        for r in range(h):
            for c in range(w):
                if grid[r][c] != petal:
                    continue
                dr = (r > cr) - (r < cr)
                dc = (c > cc) - (c < cc)
                out[cr + dr][cc + dc] = petal
    return out
```

## Generator Constraints

The generator uses square `15x15` grids. The two center rows and columns are independently sampled from `range(2, size-2)`, so centers are away from the border. For each center, up to four petals are considered in north/east/south/west order. A petal's input pixel is placed at distance at least two from the center and may land on the border; its output pixel is the adjacent cell in that direction. Each petal may be deleted randomly, and any endpoint collision with a previously seen center or petal endpoint deletes that petal. Generated examples are retried if an output adjacent petal cell or input endpoint would collide. Petal counts can be zero for a flower, and the two flowers can share rows or columns as long as the draw constraints pass.

## Reference Notes

ARC-DSL removes colors `3` and `7` from the input, finds the center objects of colors `2` and `1`, computes the `gravitate` unit vector from each distant petal toward its matching center, shifts petal objects by that vector, and paints them back next to the centers. The compact Code Golf solution repeatedly rotates/transposes the grid and uses a regex to move color `3`/`7` pixels inward toward their associated centers; it confirms the operation is cardinal-direction petal compaction rather than drawing lines.
