# task252 Semantics

## Sources

- Current champion builder: `solutions_py/task252.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task252.json`
- ARC-GEN task id: `a5f85a15`
- ARC-DSL task id: `a5f85a15`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a5f85a15.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a5f85a15.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task252.py`

## Pattern

The input is a square grid of size 3..15 on a black background. It contains one or more same-color diagonal line objects. Every colored cell lies on a down-right diagonal with constant `r - c`. The foreground color is any non-yellow color. The output preserves the input except that every colored cell in an odd-numbered global column is recolored yellow (`4`). Colored cells in even columns keep the original foreground color. Black cells remain black.

Equivalently: for every cell, if `grid[r][c] != 0` and `c % 2 == 1`, output `4`; otherwise output `grid[r][c]`.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value != 0 and c % 2 == 1:
                out[r][c] = 4
    return out
```

## Generator Constraints

ARC-GEN chooses a square size from 3 to 15. It samples `num_diags` from 1 to `size // 3`. Diagonals are drawn from two sets: top-edge diagonals with even nonpositive offsets `0, -2, -4, ...` and left/bottom-side diagonals with positive offsets `1..size-2`. The foreground color is a random color excluding yellow. All diagonals use the same color. The top-edge parity restriction avoids ambiguous odd/even-column behavior at the diagonal origin: every diagonal object's upper-left/start column is even, so the ARC-DSL solver's odd offsets from the object upper-left corner match global odd columns.

## Reference Notes

The ARC-DSL solver finds the diagonal objects, gets each object's upper-left corner, generates offsets `(1,1), (3,3), ..., (15,15)`, shifts those offsets by each object's upper-left corner, and fills those cells yellow. This is the same as recoloring every foreground cell at odd global column index. The Code Golf solution uses row iteration and a toggled/yellow value to implement the same column-parity recolor compactly.
