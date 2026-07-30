# task389 Semantics

## Sources

- Current champion builder: `solutions_py/task389.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task389.json`
- ARC-GEN task id: `f76d97a5`
- ARC-DSL task id: `f76d97a5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f76d97a5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f76d97a5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task389.py`

## Pattern

The input is a small `3x3` to `5x5` square grid filled with one non-gray color, with a subset of cells changed to gray `5`. The output is the same size with the gray-cell mask recolored to the original non-gray color and every other cell set to black/background `0`.

## Readable Python Solver

```python
def solve(grid):
    colors = {v for row in grid for v in row}
    color = next(v for v in colors if v != 5)
    return [[color if v == 5 else 0 for v in row] for row in grid]
```

## Generator Constraints

ARC-GEN chooses square size `3..5`. It samples between `size` and `(size*size+1)//2` gray pixels, so at least one gray cell exists and at most about half the grid is gray. The non-gray fill color is random excluding gray. The submitted NeuroGolf tensor is still the standard one-hot/padded `[1,10,30,30]`; the logical output occupies the top-left `size x size` region.

## Reference Notes

ARC-DSL extracts the two-color palette, switches the two colors, then replaces gray with black. Since the palette contains only the fill color and gray, this maps fill-color cells to gray and then black, and maps gray cells to the fill color. The Code Golf solution computes the non-gray/non-5 color from each row context and emits it exactly where the input value is gray.
