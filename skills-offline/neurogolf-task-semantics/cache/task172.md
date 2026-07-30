# task172 Semantics

## Sources

- Current champion builder: `solutions_py/task172.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task172.json`
- ARC-GEN task id: `6fa7a44f`
- ARC-DSL task id: `6fa7a44f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6fa7a44f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6fa7a44f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task172.py`

## Pattern

The input is a 3x3 colored grid. The output is the original 3 rows followed by a vertical mirror of those rows, producing a 6x3 grid. In Python terms, `output = grid + grid[::-1]`.

## Readable Python Solver

```python
def solve(grid):
    return [row[:] for row in grid] + [row[:] for row in grid[::-1]]
```

## Generator Constraints

ARC-GEN uses a fixed square size of 3. Each of the 9 cells is sampled from a random set of four nonzero colors, with repeats allowed. The output height is exactly 6 and width exactly 3.

## Reference Notes

The ARC-DSL solver computes `hmirror(I)` and vertically concatenates it below the original grid. The Code Golf solution is exactly `g + g[::-1]`. All sources agree.
