# task111 Semantics

## Sources

- Current champion builder: `solutions_py/task111.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task111.json`
- ARC-GEN task id: `48d8fb45`
- ARC-DSL task id: `48d8fb45`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_48d8fb45.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_48d8fb45.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task111.py`

## Pattern

The logical input is a `10x10` grid containing several separated `3x3` sprites
in one non-gray color. A single gray marker (`5`) sits directly above the target
sprite's center column. The output is the target sprite's `3x3` crop only,
padded by NeuroGolf to `[1,10,30,30]`.

If the gray marker is at `(mr, mc)`, the target crop starts at
`(mr + 1, mc - 1)` and has shape `3x3`.

## Readable Python Solver

```python
def solve(grid):
    marker = None
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 5:
                marker = (r, c)
                break
        if marker is not None:
            break
    mr, mc = marker
    r0, c0 = mr + 1, mc - 1
    return [row[c0:c0 + 3] for row in grid[r0:r0 + 3]]
```

## Generator Constraints

- Input size is fixed at `10x10`; output crop is fixed at `3x3`.
- There are up to four non-overlapping sprite placements; generator rejection
  keeps sprites separated enough to identify the marked one.
- Every sprite includes its center-top cell `(0,1)` and is diagonally connected.
- The marker is always gray and is placed at `target_row - 1, target_col + 1`.
- The sprite color is nonzero and not gray.

## Reference Notes

ARC-DSL finds the singleton gray marker, extracts the adjacent object, and returns its subgrid. The Code Golf solution flattens the grid, finds gray `5`, and slices three rows offset from that index.
