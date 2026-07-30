# task175 Semantics

## Sources

- Current champion builder: `solutions_py/task175.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task175.json`
- ARC-GEN task id: `73251a56`
- ARC-DSL task id: `73251a56`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_73251a56.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_73251a56.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task175.py`

## Pattern

The input is a 21x21 patterned grid with several black rectangular cutouts. The underlying pattern is symmetric across the main diagonal. Each non-diagonal cell pair `(r, c)` and `(c, r)` has the same true color, and the generator guarantees that at least one cell of every off-diagonal mirrored pair remains visible. Diagonal cells all have one repeated diagonal color, but diagonal cutouts can leave zeros because a diagonal cell mirrors to itself.

The output restores the full pattern: fill every black cutout from its diagonal mirror when possible, then fill any remaining diagonal zeros with the diagonal color.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    merged = [[max(grid[r][c], grid[c][r]) for c in range(n)] for r in range(n)]

    # Remaining zeros can only be diagonal cells whose mirror is also zero.
    counts = {}
    for row in merged:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    fill = max(counts, key=counts.get)
    for r in range(n):
        for c in range(n):
            if merged[r][c] == 0:
                merged[r][c] = fill

    diag = merged[0][0]
    for i in range(n):
        merged[i][i] = diag
    return merged
```

## Generator Constraints

The grid is always square, normally size 21 in generated data. The hidden full grid is built from integer ratios of row/column coordinates and then color-wrapped with `mod` in 5..9 and `modset` in 1..4. Five random black rectangular cutouts are attempted, each width and height 2..5, though the fixed validation cases can include four rectangles. The generator rejects samples where both members of any off-diagonal mirrored pair are black. It does not reject diagonal holes, so the diagonal repair step is needed.

## Reference Notes

The ARC-DSL solver diagonal-mirrors the input, pairs each cell with its mirrored counterpart, takes elementwise maximum, replaces remaining zeros with the most common color, and then fills the main diagonal from the origin color. The Code Golf solution performs the same mirror-fill recurrence compactly over rows/columns.
