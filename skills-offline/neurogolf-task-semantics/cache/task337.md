# task337 Semantics

## Sources

- Current champion builder: `solutions_py/task337.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task337.json`
- ARC-GEN task id: `d511f180`
- ARC-DSL task id: `d511f180`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d511f180.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d511f180.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task337.py`

## Pattern

The input is a small square grid of size 3, 4, or 5. Every cell keeps its position. The only transformation is a fixed color swap: color 5 becomes color 8, color 8 becomes color 5, and every other color is unchanged.

## Readable Python Solver

```python
def solve(grid):
    out = []
    for row in grid:
        new_row = []
        for value in row:
            if value == 5:
                new_row.append(8)
            elif value == 8:
                new_row.append(5)
            else:
                new_row.append(value)
        out.append(new_row)
    return out
```

## Generator Constraints

- Grid size is sampled from 3 through 5.
- The generator fills `size * size` cells with random colors.
- Some cells are resampled specifically as gray 5 or cyan 8, making the swap visible.
- The output has the same shape as the input and no geometric changes.
- All colors outside 5 and 8 are identity-mapped.

## Reference Notes

ARC-DSL is exactly `switch(I, FIVE, EIGHT)`. ARC-GEN directly implements the same rule by changing gray to cyan and cyan to gray. The Code Golf solution is an obfuscated scalar/list recursion for the same fixed swap.
