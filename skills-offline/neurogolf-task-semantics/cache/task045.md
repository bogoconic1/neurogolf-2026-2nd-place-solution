# task045 Semantics

## Sources

- Current champion builder: `solutions_py/task045.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task045.json`
- ARC-GEN task id: `22eb0ac0`
- ARC-DSL task id: `22eb0ac0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_22eb0ac0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_22eb0ac0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task045.py`

## Pattern

The grid is a 10x10 canvas with colored endpoint pixels on the left and right
edges of several odd-numbered rows. For each active row, compare the color in
column 0 with the color in column 9. If the two endpoint colors are the same,
fill the entire row with that color. If they differ, leave the row unchanged.
All inactive rows stay background.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    for r, row in enumerate(grid):
        left = row[0]
        right = row[-1]
        if left != 0 and left == right:
            out[r] = [left] * len(row)
    return out
```

## Generator Constraints

ARC-GEN always uses a square 10x10 grid. It chooses either four or five
distinct nonzero colors and places them at rows `1, 3, 5, 7` and optionally
`9` in column 0. It also places a shuffled list of the same colors in column
9 on those same rows. Background is 0 everywhere else. Because the right edge is
a permutation of the left colors, zero never appears as an endpoint and each
left color appears exactly once on each side. Rows where the permutation fixes a
color are exactly the rows that become horizontal lines in the output.

## Reference Notes

The ARC-DSL solver partitions foreground objects, recolors each object by its color over its backdrop, filters for horizontal-line objects, and paints those objects back onto the input. This selects the same-color endpoint pair as a single horizontal line and ignores nonmatching endpoint pairs. The Code Golf solution uses the number of distinct row values as a compact test: rows with only `{0, c}` are fixed-color endpoint rows and are expanded to all `c`, while rows with `{0, a, b}` or just `{0}` remain unchanged. The references agree.
