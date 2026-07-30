# task032 Semantics

## Sources

- Current champion builder: `solutions_py/task032.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task032.json`
- ARC-GEN task id: `1e0a9b12`
- ARC-DSL task id: `1e0a9b12`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1e0a9b12.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1e0a9b12.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task032.py`

## Pattern

The input is a square grid of size 4, 5, or 6. Each column has at most one nonzero color, possibly repeated in several rows, with zeros elsewhere. The output independently packs each column downward: all zeros move to the top of the column and the nonzero cells of that column occupy the bottom rows. Since every column has only one nonzero color, this is equivalent to sorting each column numerically ascending or applying gravity downward within each column.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        vals = [grid[r][c] for r in range(h) if grid[r][c] != 0]
        start = h - len(vals)
        for i, value in enumerate(vals):
            out[start + i][c] = value
    return out
```

A shorter equivalent for these generator constraints is `list(map(list, zip(*map(sorted, zip(*grid)))))`.

## Generator Constraints

ARC-GEN chooses `size` in `4..6`, samples `size` distinct colors, and for each column samples a count in `0..ceil(size/2)-1`. It then chooses that many active source rows for the column. The total number of active cells is at least `size`. Each column's active cells all share that column's color. Some sampled colors may be `0` in the fixed validation examples, which simply means an empty-looking column; the transformation still behaves as column sorting/gravity. Output shape is the same square size as input.

## Reference Notes

ARC-DSL rotates the input, sorts each row, then rotates back, which is column-wise sorting. Code Golf 2025 does the same by transposing, sorting each transposed row, and transposing back. There is no cross-column interaction, color remapping, or tie-breaking beyond ordinary sorting of zeros before nonzero values.
