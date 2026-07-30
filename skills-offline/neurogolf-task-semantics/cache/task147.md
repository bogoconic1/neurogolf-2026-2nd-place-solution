# task147 Semantics

## Sources

- Current champion builder: `solutions_py/task147.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task147.json`
- ARC-GEN task id: `67385a82`
- ARC-DSL task id: `67385a82`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_67385a82.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_67385a82.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task147.py`

## Pattern

The input is a small black grid containing green (`3`) cells. The output keeps isolated green cells green, recolors every green cell that has at least one 4-neighbor green cell to cyan (`8`), and leaves black background unchanged. In object terms, size-1 green connected components stay green; every green component of size greater than 1 is recolored cyan.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 3:
                continue
            has_friend = False
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 3:
                    has_friend = True
                    break
            out[r][c] = 8 if has_friend else 3
    return out
```

## Generator Constraints

The generated grid width and height are independently sampled from 3 through 6. A random subset of pixels is colored green in the input; the rest are black. The output is initially the same shape with each input green cell recolored cyan, then ARC-GEN `edgefree_pixels` restores any input green cell with zero edge-adjacent nonzero neighbors back to green. Out-of-bounds neighbors count as background. There are no other colors in the generated inputs or outputs.

## Reference Notes

ARC-DSL finds all green objects, separates size-one objects, and fills the non-singleton green objects with cyan. Code Golf uses repeated transposed neighbor propagation to distinguish isolated green pixels from green pixels with edge neighbors. All references agree that the test is 4-neighbor adjacency, not diagonal adjacency.
