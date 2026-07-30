# task061 Semantics

## Sources

- Current champion builder: `solutions_py/task061.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task061.json`
- ARC-GEN task id: `29ec7d0e`
- ARC-DSL task id: `29ec7d0e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_29ec7d0e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_29ec7d0e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task061.py`

## Pattern

The active grid is fixed at 18x18. Before cutouts, every cell is a multiplication table value

```python
value(r, c) = (r * c) % P + 1
```

where `P` is an integer period/modulus in `4..9`. The input is this colored table with five black rectangular cutouts. The output restores the complete table, replacing every black cutout cell with the correct periodic value. Padded space outside the 18x18 active grid remains empty in the NeuroGolf tensor.

## Readable Python Solver

```python
def solve(grid):
    n = 18
    visible = {grid[r][c] for r in range(n) for c in range(n) if grid[r][c] != 0}
    P = max(visible)
    return [[(r * c) % P + 1 for c in range(n)] for r in range(n)]
```

## Generator Constraints

The generator fixes `size=18`. It samples `mod` uniformly from `4..9`, fills the full table with `(row * col) % mod + 1`, then places five black rectangular cutouts. Each cutout has width and height in `2..4` and is positioned inside the 18x18 grid. The output is the original uncut table. The generated examples preserve enough visible colors to infer the period from global color presence.

## Reference Notes

The ARC-DSL solver removes black objects, inspects the rightmost column and bottom row to recover vertical and horizontal periods, then paints shifted copies of the non-black cells at period offsets. The Code Golf solution uses the generator's bounded family directly: infer `P` as `max(grid[-1])`/visible max and recompute `(x*y) % P + 1` for all 18x18 cells.
