# task151 Semantics

## Sources

- Current champion builder: `solutions_py/task151.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task151.json`
- ARC-GEN task id: `67a423a3`
- ARC-DSL task id: `67a423a3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_67a423a3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_67a423a3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task151.py`

## Pattern

The input is a square grid of size 4 through 12. It contains a full horizontal line in one non-yellow color and a full vertical line in a different non-yellow color. The line intersection is the only cell where both generated lines meet, and its visible color is the horizontal-line color because the generator writes the vertical line first and then overwrites the row. The output preserves the two lines and fills the eight neighboring cells around the intersection with yellow (`4`), skipping the center cell itself. A random `xpose` branch may transpose both input and output, but the visible rule is still: find the crossing of the two full lines and paint its 8-neighborhood yellow.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g: exec('g[::-1]=eval(re.sub("0, [^0](?=[^(]*\\([^0])","4,4",f"{*zip(*g),}"));' * 4) or g


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN samples `size` uniformly from 4 through 12. The crossing row and column are interior coordinates in `1..size-2`, so the full 3x3 neighborhood around the intersection is always inside the grid. The two line colors are random distinct colors excluding yellow (`4`). The vertical line uses `colors[1]`; the horizontal line uses `colors[0]`; there is no background color inside the square except zeros from the padded NeuroGolf canvas outside the active grid. The final optional transpose preserves all constraints while swapping the apparent horizontal/vertical roles.

## Reference Notes

ARC-DSL solves this as `leastcolor -> colorfilter -> merge -> delta -> first -> neighbors -> fill(I, FOUR, neighbors)`. The least frequent visible color is the intersection object under the DSL representation, and `neighbors` gives the eight surrounding cells. The Code Golf solution repeatedly transposes/string-rewrites to replace cells neighboring the crossing with `4`, agreeing with the 8-neighborhood fill rule.
