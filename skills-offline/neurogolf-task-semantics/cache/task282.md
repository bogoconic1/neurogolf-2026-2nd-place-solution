# task282 Semantics

## Sources

- Current champion builder: `solutions_py/task282.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task282.json`
- ARC-GEN task id: `b60334d2`
- ARC-DSL task id: `b60334d2`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b60334d2.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b60334d2.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task282.py`

## Pattern

The input is a 9x9 grid with black background (`0`) and either three or four isolated gray seed pixels (`5`). Around each gray seed, the output writes a 3x3 neighborhood pattern: the seed cell itself becomes black, the four axial neighbors become blue (`1`), and the four diagonal neighbors become gray (`5`). All other cells stay black. Seed centers are sampled away from the outer border and their 3x3 neighborhoods are guaranteed not to overlap.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 5:
                continue
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                out[r + dr][c + dc] = 1
            for dr, dc in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
                out[r + dr][c + dc] = 5
    return out
```

## Generator Constraints

The raw grid size is fixed at `9x9`. There are `3` or `4` gray seed pixels. Each seed row and column is chosen from `1..7`, so its full 3x3 neighborhood fits inside the grid. The generator rejects placements whose 3x3 boxes overlap, so outputs from different seeds never conflict or require tie-breaking. The only meaningful input colors are black `0` and gray `5`; output colors are black `0`, blue `1`, and gray `5`.

## Reference Notes

The ARC-DSL solver finds all gray cells, zeros the original grid, fills direct neighbors (`dneighbors`) with blue, and fills diagonal neighbors (`ineighbors`) with gray. The Code Golf solution is obfuscated but follows the same local-neighbor expansion. There is no object identity, ordering, or global geometry beyond the non-overlap guarantee.
