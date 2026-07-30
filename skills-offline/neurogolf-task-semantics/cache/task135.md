# task135 Semantics

## Sources

- Current champion builder: `solutions_py/task135.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task135.json`
- ARC-GEN task id: `5bd6f4ac`
- ARC-DSL task id: `5bd6f4ac`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5bd6f4ac.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5bd6f4ac.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task135.py`

## Pattern

The input is a `9 x 9` grid, interpreted as a `3 x 3` arrangement of `3 x 3` blocks. The output is the top-right `3 x 3` block: rows `0..2` and columns `6..8` of the input. Colors are arbitrary digits `0..9`; no object semantics or tie-breaking are involved.

## Readable Python Solver

```python
def solve(grid):
    return [row[6:9] for row in grid[:3]]
```

## Generator Constraints

ARC-GEN uses fixed `size=3`, so every generated input is `9 x 9` and every output is `3 x 3`. Each of the 81 input cells is sampled independently as either black or a random nonzero color. The task output always copies the fixed top-right crop; it does not depend on color counts or geometry.

## Reference Notes

The ARC-DSL solver is `crop(I, tojvec(SIX), THREE_BY_THREE)`, which is a crop starting at `(0, 6)` of size `3 x 3`. The Code Golf 2025 solution is `lambda g:[r[6:] for r in g[:3]]`, agreeing exactly.
