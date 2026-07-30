# task220 Semantics

## Sources

- Current champion builder: `solutions_py/task220.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task220.json`
- ARC-GEN task id: `913fb3ed`
- ARC-DSL task id: `913fb3ed`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_913fb3ed.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_913fb3ed.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task220.py`

## Pattern

The input is a square black grid containing one to three isolated seed pixels.
The possible seed colors are 2, 3, and 8. Each seed paints its eight neighboring
cells as a 3x3 halo while the center seed pixel keeps its original color. The
halo color is determined by the seed color: `2 -> 1`, `3 -> 6`, and `8 -> 4`.
All other cells remain black. The output grid has the same height and width as
the input; in NeuroGolf it is embedded in the standard dense `[1, 10, 30, 30]`
output with zero-hot padding outside the original grid.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    halo = {2: 1, 3: 6, 8: 4}
    seeds = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] in halo:
                seeds.append((r, c, grid[r][c]))
    for r, c, color in seeds:
        fill = halo[color]
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr][cc] = fill
        out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN samples a square size from 6 to 16. It chooses a nonempty subset of the
colors `[2, 3, 8]`, so there are one, two, or three seed pixels and at most one
seed of each color. Seed rows and columns are sampled from `1..size-2`, so every
3x3 halo fits inside the active grid. The generator rejects overlapping 3x3
neighborhoods, so halos never conflict and there is no tie-breaking ambiguity.
The input is otherwise black.

## Reference Notes

ARC-DSL gets the positions of colors 3, 8, and 2, expands each set with `neighbors`, then fills those neighborhoods with 6, 4, and 1 respectively. Since `neighbors` excludes the center, the original seed center remains unchanged. The Code Golf solution performs the same local transpose/rotation-style pass in a compressed form. The references and generator agree on the color map and the non-overlap invariant.
