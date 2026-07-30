# task210 Semantics

## Sources

- Current champion builder: `solutions_py/task210.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task210.json`
- ARC-GEN task id: `8be77c9e`
- ARC-DSL task id: `8be77c9e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8be77c9e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8be77c9e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task210.py`

## Pattern

The input is a fixed `3x3` grid containing a nonempty subset of blue (`1`) cells on black background. The output is the input followed by its vertical mirror, producing a `6x3` grid. In row terms, output rows are selected from input rows `[0, 1, 2, 2, 1, 0]`.

The result occupies the top-left of the standard dense NeuroGolf `[1,10,30,30]` output; all remaining rows and columns are black/zero.

## Readable Python Solver

```python
def solve(grid):
    return [row[:] for row in grid] + [row[:] for row in reversed(grid)]
```

## Generator Constraints

- Input size is fixed at `3x3`.
- At least one cell is blue.
- Only colors `0` and `1` appear.
- Output size is fixed at `6x3`.

## Reference Notes

The ARC-DSL solver is exactly `vconcat(I, hmirror(I))`; in ARC-DSL naming, `hmirror` flips across the horizontal axis, i.e. reverses row order. The Code Golf 2025 solution is `g + g[::-1]`, matching the generator.
