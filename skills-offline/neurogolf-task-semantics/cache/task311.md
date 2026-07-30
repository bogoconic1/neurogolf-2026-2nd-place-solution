# task311 Semantics

## Sources

- Current champion builder: `solutions_py/task311.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task311.json`
- ARC-GEN task id: `c9e6f938`
- ARC-DSL task id: `c9e6f938`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c9e6f938.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c9e6f938.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task311.py`

## Pattern

The input is a `3 x 3` grid containing some orange pixels on black background.
The output is `3 x 6`: each row is concatenated with its horizontal reverse,
so `row -> row + row[::-1]`. In the NeuroGolf tensor, this lives in the
upper-left of the standard `[1,10,30,30]` output canvas and the rest is zero.

## Readable Python Solver

```python
def solve(grid):
    return [row + row[::-1] for row in grid]
```

## Generator Constraints

- The generated semantic grid size is fixed at `3 x 3`.
- At least one pixel is present.
- Present pixels are orange; all other cells are black.
- Output width is exactly `2 * size = 6` and height remains 3.
- Each present input pixel at `(r, c)` appears at `(r, c)` and `(r, 5 - c)` in
  the output.

## Reference Notes

ARC-GEN gives the fixed mirror rule. ARC-DSL solves it as `hconcat(I, vmirror(I))`. The Code Golf solution is the same row concatenation with a Python reverse. There is no ambiguity or tie-breaking.
