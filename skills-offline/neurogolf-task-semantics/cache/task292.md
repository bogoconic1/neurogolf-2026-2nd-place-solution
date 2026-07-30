# task292 Semantics

## Sources

- Current champion builder: `solutions_py/task292.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task292.json`
- ARC-GEN task id: `ba26e723`
- ARC-DSL task id: `ba26e723`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ba26e723.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ba26e723.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task292.py`

## Pattern

The visible input grid has height 3 and width 10 through 20. It contains a yellow (`4`) zigzag: row 1 is yellow in every column, and each column also has one yellow cell in row 0 or row 2 depending on `(column + flip) % 2`. Background is black (`0`).

The output keeps the same support, but recolors yellow cells in columns divisible by 3 to magenta (`6`). Yellow cells in all other columns stay yellow. Background remains black and padded area remains zero-hot in the NeuroGolf tensor.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == 4 and c % 3 == 0:
                out[r][c] = 6
    return out
```

## Generator Constraints

ARC-GEN fixes the visible height to 3 and chooses width uniformly from 10 through 20. The `flip` bit controls whether the extra yellow cell in each column is on row 0 or row 2 for even/odd columns. Every column has exactly two yellow cells: the middle row and one alternating outer row. All foreground is color 4 in the input. The output uses color 6 exactly on columns `0, 3, 6, 9, 12, 15, 18` that exist within the chosen width.

Important edge cases:

- width can end before some modulo-3 marked columns, but the standard tensor still has 30 columns
- padded columns beyond the visible width are zero-hot, not background channel 0
- both flip parities must keep the same modulo-3 recolor rule
- no other colors appear in the input

## Reference Notes

ARC-DSL filters color-4 cells by `column == 3 * (column // 3)` and fills those cells with color 6. ARC-GEN makes the fixed support and modulo-3 rule explicit. The Code Golf solution encodes the same modulo pattern tersely.
