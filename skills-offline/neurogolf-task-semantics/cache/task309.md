# task309 Semantics

## Sources

- Current champion builder: `solutions_py/task309.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task309.json`
- ARC-GEN task id: `c8f0f002`
- ARC-DSL task id: `c8f0f002`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c8f0f002.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c8f0f002.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task309.py`

## Pattern

The input is a 3-row grid of width 4..6 using only colors 1, 7, and 8. The
output is the same shape with every color 7 cell recolored to 5. Colors 1 and 8
stay unchanged, and the padded zero area remains zero.

## Readable Python Solver

```python
def solve(grid):
    return [[5 if x == 7 else x for x in row] for row in grid]
```

## Generator Constraints

- Height is fixed at 3.
- Width is `4..6`.
- Every active cell is one of colors `(1, 7, 8)`.
- At least the generated active grid is fully filled; padding outside the task
  rectangle is black/zero in the NeuroGolf tensor.
- The only transformation is `7 -> 5`.

## Reference Notes

ARC-GEN gives the fixed recolor rule. ARC-DSL is exactly `replace(I, 7, 5)`. The Code Golf solution implements the same recursive replacement. No disagreement was found.
