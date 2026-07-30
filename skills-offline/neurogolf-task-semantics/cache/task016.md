# task016 Semantics

## Sources

- Current champion builder: `solutions_py/task016.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task016.json`
- ARC-GEN task id: `0d3d703e`
- ARC-DSL task id: `0d3d703e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_0d3d703e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_0d3d703e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task016.py`

## Pattern

The input is a 3x3 grid whose rows are constant-color stripes derived from three sampled colors. The output recolors every cell by a fixed involutive color mapping: `1 <-> 5`, `2 <-> 6`, `3 <-> 4`, `8 <-> 9`, while `0` and `7` are unchanged. Color orange (`7`) is excluded from generated colors, but the mapping leaves it fixed.

## Readable Python Solver

```python
def solve(grid):
    mapping = {0: 0, 1: 5, 2: 6, 3: 4, 4: 3, 5: 1, 6: 2, 7: 7, 8: 9, 9: 8}
    return [[mapping[v] for v in row] for row in grid]
```

## Generator Constraints

- Grid size is fixed at 3x3.
- The generator samples three distinct colors, excluding orange/color 7.
- It forms a simple 3x3 stripe grid from those colors and transposes it.
- The output applies the fixed colormap `[0,5,6,4,3,1,2,7,9,8]` to every cell.

## Reference Notes

- ARC-DSL implements the same mapping as four pair swaps: `3<->4`, `8<->9`, `2<->6`, and `1<->5`.
- ARC-GEN gives the full colormap directly.
- The Code Golf solution uses a compact arithmetic formula equivalent to the same color permutation on scalar labels.
