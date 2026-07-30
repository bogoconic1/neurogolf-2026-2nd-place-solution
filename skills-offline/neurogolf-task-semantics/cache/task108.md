# task108 Semantics

## Sources

- Current champion builder: `solutions_py/task108.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task108.json`
- ARC-GEN task id: `46f33fce`
- ARC-DSL task id: `46f33fce`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_46f33fce.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_46f33fce.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task108.py`

## Pattern

The logical ARC input is a `10x10` grid. Only cells whose row and column are
both odd can be non-black: coordinates `(2*r + 1, 2*c + 1)` for
`r,c in 0..4`. The generator places five or six colored pixels on those odd
coordinates.

The logical ARC output is `20x20`. Each odd-coordinate input pixel at
`(2*r + 1, 2*c + 1)` becomes a solid `4x4` block at output rows
`4*r..4*r+3` and columns `4*c..4*c+3`. Empty sampled positions become black
`4x4` blocks. The NeuroGolf tensor still uses the standard `[1, 10, 30, 30]`
one-hot canvas, so output rows or columns `20..29` are blank/zero outside the
logical `20x20` result.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(20)] for _ in range(20)]
    for r in range(5):
        for c in range(5):
            color = grid[2 * r + 1][2 * c + 1]
            for dr in range(4):
                for dc in range(4):
                    out[4 * r + dr][4 * c + dc] = color
    return out
```

## Generator Constraints

- Input size is fixed at `10x10`.
- The generator samples five or six distinct positions from the `5x5` odd-cell
  lattice.
- Sampled colors are random ARC colors; repeated colors are allowed.
- Even rows, even columns, and unsampled odd lattice cells are black.
- Output size is fixed at `20x20` before NeuroGolf padding to `30x30`.
- There are no shape branches, tie-breaks, or object interactions beyond
  independent odd-cell sampling and block replication.

## Reference Notes

ARC-DSL expresses the same operation as rotate, downscale by two, rotate back, then upscale by four. For this generator that is equivalent to sampling the odd odd-coordinate lattice and writing `4x4` blocks.

The Code Golf 2025 solution is a recursive/list-golf expression that indexes the odd sampled rows and columns after repetition; it agrees with the fixed sample-and-tile interpretation.
