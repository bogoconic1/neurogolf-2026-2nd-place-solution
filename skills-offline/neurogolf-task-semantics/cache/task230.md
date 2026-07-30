# task230 Semantics

## Sources

- Current champion builder: `solutions_py/task230.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task230.json`
- ARC-GEN task id: `95990924`
- ARC-DSL task id: `95990924`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_95990924.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_95990924.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task230.py`

## Pattern

The input is a square black grid of size 10 or 15 containing several non-overlapping 2x2 gray blocks. The output keeps the gray blocks unchanged and colors the four diagonal cells just outside each block's 4x4 outbox: upper-left corner becomes color 1, upper-right color 2, lower-left color 3, and lower-right color 4. All other cells stay black.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h - 1):
        for c in range(w - 1):
            if grid[r][c] == grid[r][c + 1] == grid[r + 1][c] == grid[r + 1][c + 1] == 5:
                out[r - 1][c - 1] = 1
                out[r - 1][c + 2] = 2
                out[r + 2][c - 1] = 3
                out[r + 2][c + 2] = 4
    return out
```

## Generator Constraints

- Grid size is `5 * factor`, where factor is 2 or 3, so sizes are 10 or 15.
- Each object is exactly a 2x2 gray block.
- Block top-left rows and columns are sampled from `1..size-4`, leaving one-cell room around the block for the colored corners.
- Blocks are non-overlapping with a 4x4 exclusion length in the generator's overlap check.
- The number of attempted block placements is `3 * factor`; skipped overlaps can reduce the final count.

## Reference Notes

- ARC-DSL finds objects, takes each outbox, and fills its four corners with colors 1, 2, 3, and 4.
- The Code Golf 2025 solution uses a compact local hash over neighboring rows to classify each cell.
