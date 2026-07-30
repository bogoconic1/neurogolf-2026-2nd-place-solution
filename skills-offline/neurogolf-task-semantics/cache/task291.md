# task291 Semantics

## Sources

- Current champion builder: `solutions_py/task291.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task291.json`
- ARC-GEN task id: `b9b7f026`
- ARC-DSL task id: `b9b7f026`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b9b7f026.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b9b7f026.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task291.py`

## Pattern

The input is a 12..18 by 12..18 grid containing 4 to 7 non-overlapping solid colored rectangles on a black background. The first generated rectangle has an internal black rectangular hole cut out of it, making it a colored donut-like object. All other rectangles are solid.

The output is a 1x1 grid whose single color is the color of the rectangle that contains or frames the black hole. In ARC terms, find the smallest black object, find the non-black object adjacent to it, and output that object color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, i=1: 4 % len({r.count(i) for r in g}) * [[i]] or p(g, i + 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN chooses 4 to 7 non-overlapping rectangles in a grid whose width and height are each 12 through 18. Rectangle widths and heights are 2 through 8, but the first rectangle is forced to be at least 3 by 3 so it can contain a hole. The hole is a black rectangle strictly inside the first colored rectangle, with positive width and height and at least a one-cell colored border around it. The output is always the first rectangle color as a 1x1 grid.

Important edge cases:

- the hole can be wider or taller than one cell
- the first rectangle can be any non-background color
- other rectangles may have arbitrary colors and sizes but do not overlap the donut rectangle
- the visible grid is padded to the standard NeuroGolf tensor, while the logical output is a 1x1 color

## Reference Notes

ARC-DSL finds objects, selects the minimum-size object, finds the object adjacent to it, and returns that object color as a 1x1 canvas. Since the only enclosed non-border black object is the hole and it is smaller than the outside background, this identifies the donut rectangle.

The Code Golf solution increments candidate colors until row-count signatures indicate the holed rectangle color. It exploits that the donut color has nonuniform row counts because of the missing black hole.
