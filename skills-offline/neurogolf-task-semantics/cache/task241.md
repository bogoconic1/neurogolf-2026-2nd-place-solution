# task241 Semantics

## Sources

- Current champion builder: `solutions_py/task241.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task241.json`
- ARC-GEN task id: `9dfd6313`
- ARC-DSL task id: `9dfd6313`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_9dfd6313.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_9dfd6313.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task241.py`

## Pattern

The task is a square-grid diagonal mirror, equivalent to transposing the grid. Gray cells lie on the main diagonal and remain fixed. Colored foreground cells appear below the diagonal in the input and are moved to the symmetric positions above the diagonal in the output. Background stays background.

## Readable Python Solver

```python
def solve(grid):
    return [list(row) for row in zip(*grid)]
```

## Generator Constraints

ARC-GEN uses square grids with size 3..9. The main diagonal is gray. Optional colored cells are placed only where row > column, using colors sampled from 1..9 excluding gray. The output places each such colored cell at the transposed coordinate. There can be zero or many lower-triangle colored cells, with repeated colors allowed.

## Reference Notes

ARC-DSL returns `dmirror(I)`, and the Code Golf 2025 solution is exactly `zip(*g)`.
