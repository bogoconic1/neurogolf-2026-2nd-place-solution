# task373 Semantics

## Sources

- Current champion builder: `solutions_py/task373.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task373.json`
- ARC-GEN task id: `e9afcf9a`
- ARC-DSL task id: `e9afcf9a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e9afcf9a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e9afcf9a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task373.py`

## Pattern

The input is a 2x6 grid with two row colors: every cell in row 0 has color `a`, and every cell in row 1 has color `b`. The output is the same 2x6 size but alternates those two colors by column. Even columns keep the input row order `[a, b]`; odd columns swap it to `[b, a]`.

Equivalently:

```python
output[0] = [a, b, a, b, a, b]
output[1] = [b, a, b, a, b, a]
```

## Readable Python Solver

```python
def solve(grid):
    a, b = grid[0][0], grid[1][0]
    return [[a, b, a, b, a, b], [b, a, b, a, b, a]]
```

## Generator Constraints

- Width is fixed at 6 and height is fixed at 2.
- `colors` are two distinct random ARC colors in random order.
- Input row 0 is entirely `colors[0]`; input row 1 is entirely `colors[1]`.
- Output alternates by `(r + c) % 2`, placing `colors[r]` into that parity row.
- No background/zero special case is generated unless one of the random colors is black, which `random_colors(2)` normally avoids.

## Reference Notes

ARC-DSL crops the first column, mirrors it vertically to swap the two colors, then horizontally concatenates the original and swapped columns into a 2-column tile and repeats it to width 6. Code Golf 2025 expresses the same by taking one input column tuple, repeating it three times for the first output row, and reversing it for the second row.
