# task119 Semantics

## Sources

- Current champion builder: `solutions_py/task119.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task119.json`
- ARC-GEN task id: `508bd3b6`
- ARC-DSL task id: `508bd3b6`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_508bd3b6.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_508bd3b6.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task119.py`

## Pattern

The grid is always a 12x12 square. One side of the grid contains a solid red wall of thickness `depth` (color 2), after an optional gravity rotation/reflection. A diagonal bouncing path touches or approaches that wall. The first 2 or 3 cells of the path are visible as cyan (color 8). The remaining path cells are green in the true output but are hidden as black (color 0) in the input. The output restores those hidden green cells (color 3), preserving the cyan prefix and the red wall.

The path is a V-shaped absolute-value diagonal before gravity/flip: for each column `c`, the path row is `depth + abs(mid - c)`. The generator optionally flips horizontally, then applies gravity to move the red wall to top, left, bottom, or right. Therefore the visible cyan segment plus the wall side/thickness determine the full path and which black cells must become green.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g: exec('g[::-1]=zip(*eval(re.sub("0(?=.{40}[38].{40}[238])","3",str(g))));' * 40) or g


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Grid size is fixed at 12x12.
- `depth` is 1..5.
- `mid` is between `depth` and `size - depth - 2`, so the V path stays within the grid.
- `shown` is 2 or 3; only the first visible path cells remain cyan in the input.
- `flip` optionally mirrors the canonical grid horizontally before gravity.
- `gravity` moves the wall/path orientation to any of the four sides.
- Colors are black 0, red wall 2, hidden green path 3, and visible cyan prefix 8.
- The input is the output with all green cells replaced by black.

## Reference Notes

ARC-DSL identifies the small cyan object and large red wall object, chooses the cyan corner that indicates path direction, paints a long green diagonal, repaints the wall, then finds the path segment adjacent to the wall and paints the opposite diagonal before restoring cyan and red objects. This confirms that the hidden path can bend/rebound and that red/cyan must be preserved after reconstruction.

The Code Golf solution repeatedly applies a regex over the grid and its transpose/reversal to turn black cells into green when they lie on the inferred diagonal relation with nearby cyan/red/green evidence.
