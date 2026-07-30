# task125 Semantics

## Sources

- Current champion builder: `solutions_py/task125.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task125.json`
- ARC-GEN task id: `543a7ed5`
- ARC-DSL task id: `543a7ed5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_543a7ed5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_543a7ed5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task125.py`

## Pattern

The input is a `15 x 15` grid with cyan background (`8`). It contains several solid pink (`6`) rectangles and smaller yellow (`4`) rectangles cut out inside some of the pink rectangles. The output preserves the input objects and changes the area around them:

- every pink rectangle gets a one-cell-thick green (`3`) outer border around its bounding box;
- every yellow rectangle remains yellow and its one-cell surrounding ring inside the containing pink rectangle becomes yellow as well;
- cyan background remains cyan except where a pink rectangle's outer border is painted green.

In effect, pink objects are expanded outward with color `3`, while yellow sub-objects are expanded outward with color `4`. The original pink and yellow interiors keep their colors, so yellow expansion overwrites pink/green only in the one-cell delta around the yellow rectangle.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g: exec('g[::-1]=zip(*eval(re.sub("[38](?=(..(4|6).{40}|.{49})[46])",r"4-1\\2%2",str(g))));' * 20) or g


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN uses a fixed square size of `15`. The background is cyan (`8`). It samples three non-overlapping pink rectangles with widths and heights in `2..7`, placed at least one cell away from the canvas border. The rectangles are separated by at least two cells, so their one-cell outboxes do not collide.

For each pink rectangle, the generator may add one interior yellow rectangle. The yellow rectangle has positive width and height strictly smaller than the pink rectangle and is placed strictly inside it, leaving at least one pink cell between the yellow rectangle and the pink rectangle boundary. The generator requires the total yellow area to be at least `2 * boxes`, so there is useful yellow structure in every generated puzzle. Hand-authored examples may include yellow rectangles of width or height `1`.

The output starts from a cyan canvas, paints each pink rectangle's one-cell outbox green (`3`), then paints the pink interior, then paints each yellow rectangle and its one-cell delta yellow (`4`).

## Reference Notes

The ARC-DSL solver extracts objects, filters the pink (`6`) objects, fills their `outbox` with green (`3`), then extracts each object's `delta` and fills that with yellow (`4`). Because yellow rectangles are holes/subobjects inside pink rectangles, this matches the generator's intended overwrite order: green around pink boxes first, yellow expansion around yellow boxes second.

The Code Golf solution repeatedly rotates the grid and applies a regex-like local substitution around colors `4` and `6`, which reinforces that the task is local boundary expansion around rectangles rather than global counting or object selection.
