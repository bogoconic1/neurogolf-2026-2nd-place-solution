# task329 Semantics

## Sources

- Current champion builder: `solutions_py/task329.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task329.json`
- ARC-GEN task id: `d23f8c26`
- ARC-DSL task id: `d23f8c26`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d23f8c26.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d23f8c26.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task329.py`

## Pattern

The input is an odd-sized square grid. The output has the same size and keeps
only the center column of the input. Every cell outside the center column is set
to black (`0`). The colors in the center column are copied unchanged, including
black cells.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    center = w // 2
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        out[r][center] = grid[r][center]
    return out
```

## Generator Constraints

ARC-GEN chooses odd square sizes `3`, `5`, `7`, or `9`. Cell colors are sampled
from black (`0`) and random non-black colors until the grid uses more than two
distinct colors. The output size equals the input size. Because the size is odd,
the center column is unambiguous at `size // 2`.

## Reference Notes

The ARC-DSL solver computes all indices, halves the grid width, filters indices whose last coordinate equals the half-width, and fills all other positions with black. The Code Golf 2025 solution is the same rule in compact list form: construct rows with zeros on both sides and the original center-column value in the middle. There is no ambiguity across references.
