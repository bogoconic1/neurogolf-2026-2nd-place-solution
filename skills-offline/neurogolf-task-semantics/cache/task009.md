# task009 Semantics

## Sources

- Current champion builder: `solutions_py/task009.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task009.json`
- ARC-GEN task id: `06df4c85`
- ARC-DSL task id: `06df4c85`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_06df4c85.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_06df4c85.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task009.py`

## Pattern

The input is a square logical bitmap drawn as a line grid: each logical cell expands to a 2x2 block and grid lines use a fixed line color. Some colored endpoint dots are shown in the logical cells. For each color, if two endpoints of that color share a logical row or share a logical column, draw the straight horizontal or vertical segment connecting them in the logical bitmap. Then render the completed logical bitmap back through the same line-grid format, preserving the line color and black empty cells.

The generator always includes one L-shaped group made from three corners of a rectangle, all with the same color, so two perpendicular same-color segments must be completed. It may also include up to two additional same-color endpoint pairs forming extra horizontal or vertical lines, and one random distractor dot that must not connect to another dot of the same color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, *r, i=0: [x | max({*g[i::3]} & {*g[:(i := (i + 1))]}) for x in r] or [*map(p, g, *map(p, zip(*g), *g))]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Logical size is square, between 6 and 10 inclusive. The rendered line grid has side length `3 * size`.
- `linecolor` is any nonzero color and is excluded from endpoint colors.
- One primary L shape is created from three of the four corners of a rectangle of width and height 3 to `size-2`; all three points share one color.
- Up to two extra non-overlapping straight endpoint pairs may be added with distinct colors.
- One random distractor point is added with a color that does not match any point in the same logical row or column.
- Empty logical cells are black. Grid-line cells keep the line color.

## Reference Notes

ARC-DSL partitions objects, removes the background/line-color object, connects same-color pairs that share row or column, paints those lines, then fills the most common color cells back in. ARC-GEN makes the line-grid rendering explicit and confirms the fixed 2-cell block plus 1-cell gridline spacing. The Code Golf solution recursively processes rows and columns, filling cells when the same color appears on both sides in a stride-3 logical layout.
