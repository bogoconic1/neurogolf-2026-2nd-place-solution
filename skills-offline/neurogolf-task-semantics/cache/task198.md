# task198 Semantics

## Sources

- Current champion builder: `solutions_py/task198.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task198.json`
- ARC-GEN task id: `83302e8f`
- ARC-DSL task id: `83302e8f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_83302e8f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_83302e8f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task198.py`

## Pattern

The input is a colored linegrid. The line color is a random color other than yellow (`4`) or green (`3`). The cell interiors are black. Some black pixels are also punched through the colored grid lines; these holes connect neighboring black cell interiors into larger non-square black components.

The output keeps the colored grid-line pixels unchanged. Every black connected component is recolored by shape: components that are perfect squares become green (`3`), while every non-square black component becomes yellow (`4`). The punched line pixels themselves are part of the non-square components and become yellow.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g: exec("g[::-1]=zip(*eval(re.sub('(?=0|3, 4|3[^)]*[^)34]{6})','6^5-',str(g))));" * 24) or g


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- `minisize` is `3..5`.
- `size` is `randint(8,10) - minisize`, so the number of large cells is typically `3..7` depending on `minisize`.
- The full linegrid side length is `size * (minisize + 1) - 1`, at most `29`, so it fits within the NeuroGolf 30x30 canvas.
- The line color is random but excludes yellow (`4`) and green (`3`).
- Black hole pixels are sampled only from grid-line positions where either coordinate is congruent to `minisize` modulo `minisize+1`.
- The number of holes is between `size + minisize` and `size * minisize`.
- A hole on a horizontal line connects the cells above and below; a hole on a vertical line connects the cells left and right. Connected black components that contain such holes are non-square and should be yellow.
- Square untouched cell interiors remain isolated `minisize x minisize` black squares and should become green.

## Reference Notes

The ARC-DSL solver finds black objects, filters the square black components, recolors square components green, recolors the remaining black components yellow, and paints both sets over the original grid. This exactly matches connected-component classification by squareness.

The Code Golf solution is a compact regex/string trick that repeatedly transforms the grid, but it agrees with the same semantics: black components corresponding to intact cells become green and all black regions connected through line holes become yellow.
