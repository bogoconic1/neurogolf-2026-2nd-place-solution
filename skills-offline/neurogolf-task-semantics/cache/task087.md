# task087 Semantics

## Sources

- Current champion builder: `solutions_py/task087.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task087.json`
- ARC-GEN task id: `3c9b0459`
- ARC-DSL task id: `3c9b0459`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3c9b0459.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3c9b0459.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task087.py`

## Pattern

The input is a 3x3 grid whose cells are all drawn from exactly three non-background colors. The output is the input rotated by 180 degrees. Equivalently, reverse the row order and reverse every row.

## Readable Python Solver

```python
def solve(grid):
    return [row[::-1] for row in grid[::-1]]
```

## Generator Constraints

The generated grid is always square with size 3. The generator samples exactly three random nonzero colors and fills each of the nine cells by choosing one of those colors. There is no background in the logical 3x3 grid. In NeuroGolf tensors, the 3x3 content is embedded in the standard 30x30 one-hot input/output canvas.

## Reference Notes

ARC-DSL is exactly `rot180(I)`. The Code Golf 2025 solution is the same row-reversal and per-row reversal expression. There are no object, color-count, tie-breaking, or edge-case branches beyond the fixed 3x3 geometry.
