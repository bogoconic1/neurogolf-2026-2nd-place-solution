# task339 Semantics

## Sources

- Current champion builder: `solutions_py/task339.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task339.json`
- ARC-GEN task id: `d631b094`
- ARC-DSL task id: `d631b094`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d631b094.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d631b094.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task339.py`

## Pattern

The input is a fixed `3x3` grid containing zero background and `1..9` cells of one nonzero color. All nonzero cells have the same color. The output is a one-row horizontal bar whose length equals the number of nonzero input cells and whose color is that same nonzero color.

In the NeuroGolf dense output, this means row `0`, columns `0..count-1` are the input color, and all other cells are background.

## Readable Python Solver

```python
def solve(grid):
    vals = [v for row in grid for v in row if v != 0]
    if not vals:
        return [[]]
    return [vals[:]]
```

## Generator Constraints

- Input size is exactly `3x3`.
- Nonzero cell count is sampled from `1..9`.
- Nonzero cell positions are sampled without replacement from the 9 cells.
- The nonzero color is one random ARC color in `1..9`.
- All nonzero cells share that single color.
- The output is height `1` and width equal to the nonzero count.

## Reference Notes

ARC-DSL takes the nonzero palette color, counts the cells of that color, and returns a canvas of shape `(1, count)` filled with that color. The Code Golf solution flattens the grid and filters nonzero values, which works because all nonzero values are the same color and the output is just that many copies.
