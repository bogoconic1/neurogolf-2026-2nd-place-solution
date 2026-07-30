# task276 Semantics

## Sources

- Current champion builder: `solutions_py/task276.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task276.json`
- ARC-GEN task id: `b1948b0a`
- ARC-DSL task id: `b1948b0a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b1948b0a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b1948b0a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task276.py`

## Pattern

The input is a small `3..6` by `3..6` grid padded into NeuroGolf's one-hot tensor. Background cells are pink (`6`) and a random subset of cells are orange (`7`). The output replaces every pink/background cell with red (`2`) and leaves every orange cell unchanged.

## Readable Python Solver

```python
def solve(grid):
    return [[2 if cell == 6 else cell for cell in row] for row in grid]
```

## Generator Constraints

Random generated grids have width and height in `4..6`; ARC-AGI examples include height or width `3`. The grid starts entirely pink (`6`), then random positions are set to orange (`7`). The output grid starts red (`2`) at every cell and restores orange (`7`) at the same random positions. No other in-grid colors are generated.

## Reference Notes

ARC-DSL is exactly `replace(I, SIX, TWO)`. The Code Golf solution recursively applies the same color replacement.
