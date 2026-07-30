# task164 Semantics

## Sources

- Current champion builder: `solutions_py/task164.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task164.json`
- ARC-GEN task id: `6d0aefbc`
- ARC-DSL task id: `6d0aefbc`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6d0aefbc.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6d0aefbc.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task164.py`

## Pattern

The input is a 3x3 grid using colors 1, 6, and 8. The output keeps the input on the left and appends a vertical mirror on the right, producing a 3x6 grid. In row terms, each row becomes `row + row[::-1]`.

## Readable Python Solver

```python
def solve(grid):
    return [row + row[::-1] for row in grid]
```

## Generator Constraints

The generator fixes `size=3` and chooses each cell independently from colors `(1, 6, 8)`. Output dimensions are 3 rows by 6 columns. No background or special marker colors are used.

## Reference Notes

ARC-DSL is exactly `hconcat(I, vmirror(I))`. Code Golf 2025 uses the same row-concatenation formula.
