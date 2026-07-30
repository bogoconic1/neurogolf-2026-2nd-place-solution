# task155 Semantics

## Sources

- Current champion builder: `solutions_py/task155.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task155.json`
- ARC-GEN task id: `68b16354`
- ARC-DSL task id: `68b16354`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_68b16354.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_68b16354.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task155.py`

## Pattern

The input is a square grid of size `4..8`, filled completely with colors from the set `{1, 2, 3, 4, 7, 8}`. The output is the same grid flipped vertically: row `0` moves to the last row, row `1` moves to the second-last row, and so on. Columns and colors are unchanged. There are no objects, blanks, tie-breaks, or conditional branches beyond the variable square size.

## Readable Python Solver

```python
def solve(grid):
    return [row[:] for row in grid[::-1]]
```

## Generator Constraints

ARC-GEN chooses a square size uniformly in `4..8`. Every cell is independently selected from `color_list=(1, 2, 3, 4, 7, 8)`. The background color argument to `common.grids` is `0`, but generated cells overwrite the full `size x size` square, so valid examples contain no color `0` inside the square. NeuroGolf conversion pads the square to the standard `[1,10,30,30]` tensor with zero-hot area outside the true grid. The only edge cases are the supported sizes, especially even sizes where there is no fixed middle row and odd sizes where the middle row maps to itself.

## Reference Notes

ARC-DSL is exactly `hmirror(I)`, which in ARC terminology flips rows top-to-bottom. The Code Golf 2025 solution is `p=lambda g:g[::-1]`, the same vertical row reversal. The ARC-GEN generator writes `output[size - r - 1][c] = grid[r][c]`, confirming the row flip without color remapping or cropping.
