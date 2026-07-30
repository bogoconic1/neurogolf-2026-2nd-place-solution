# task284 Semantics

## Sources

- Current champion builder: `solutions_py/task284.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task284.json`
- ARC-GEN task id: `b7249182`
- ARC-DSL task id: `b7249182`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b7249182.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b7249182.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task284.py`

## Pattern

The input contains exactly two colored seed pixels of different colors, aligned on one row or one column. In the normalized landscape orientation the seeds are on the same row and separated by an odd span `2 * half - 1`. The output grows those seeds into two facing symmetric bracket/rail shapes. The left seed color extends rightward along the middle row, then forms a five-cell-tall vertical end with a one-cell inward cap on the top and bottom rows. The right seed color mirrors this from the right side. If the input was transposed/portrait, the same construction is applied after transposing and then transposed back.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g, n=39: -n * g or p(eval(re.sub(*['(.),.0(?=[^)]*[1-9]) \\1,\\1', '(0,.0,.)([^0]),.\\1(.{%d})' % (3 * len(g) - 13) * 2 + '(?!\\1(\\2|0)) *[\\2]*5,\\3\\2,0,\\1\\2,\\6'][n < 4].split(), f'{(*zip(*g[::-1]),)}')), n - 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

The normalized landscape grid has `half` from 4 through 10, width `2 * half + 1..4`, and height 7 through 10. The left seed column is chosen so the full two-sided shape fits horizontally, and the seed row is chosen from `1..height-6`, so all output rows `r-2..r+2` fit. Two distinct nonzero colors are chosen. Half the examples are transposed, so the same rule must work for horizontal and vertical orientations. Raw task grids can be smaller than 30x30, with the NeuroGolf graph using a no-output sentinel outside the original grid.

## Reference Notes

ARC-DSL normalizes portrait cases with `dmirror`, orders the two seed objects, connects them, finds the midpoint/center of mass, fills the left and right middle segments with the two colors, paints a two-pixel vertical center object shifted left/right, then fills the top and bottom rails before restoring orientation. The Code Golf solution repeatedly applies regex-like growth over the grid and its transpose until the two bracket halves are completed, which agrees with the generator's local mirrored fill rule.
