# task150 Semantics

## Sources

- Current champion builder: `solutions_py/task150.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task150.json`
- ARC-GEN task id: `67a3c6ac`
- ARC-DSL task id: `67a3c6ac`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_67a3c6ac.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_67a3c6ac.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task150.py`

## Pattern

The input is a square grid whose size is between 3 and 9. Cells use only colors `6`, `2`, `1`, and `7`. The output is the same grid mirrored horizontally: each row is reversed left-to-right. No colors are changed and the grid size is unchanged.

## Readable Python Solver

```python
def solve(grid):
    return [row[::-1] for row in grid]
```

## Generator Constraints

ARC-GEN samples a square size from 3 through 9. Each cell independently chooses one of the fixed colors `(6, 2, 1, 7)`. Input and output shapes are identical. There are no objects, markers, or special cases beyond horizontal reversal.

## Reference Notes

ARC-DSL is exactly `vmirror(I)`. The Code Golf solution is the row-reversal one-liner. All references agree this is a pure horizontal mirror.
