# task170 Semantics

## Sources

- Current champion builder: `solutions_py/task170.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task170.json`
- ARC-GEN task id: `6ecd11f4`
- ARC-DSL task id: `6ecd11f4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6ecd11f4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6ecd11f4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task170.py`

## Pattern

The input is a 21-28 by 21-28 black grid containing two objects:

- a large connected single-color megasprite near the top, made by scaling a 3x3 or 4x4 binary sprite by an integer factor;
- a small colored 3x3 or 4x4 color box near the bottom/right.

The large object tells which cells of the small color box should be kept. The output is the small color box with all cells cleared to zero where the corresponding downscaled large sprite cell is absent. The large object color only matters for locating the object.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(g, n=3):
    *l, = filter(int, sum(g, []))
    i = ~9 - 1 % len({*l[:-9]}) * 7
    return [[v % ~v & l[(i := (i + 1))] for *c, v in [*zip(*g, r)][::n] if [c[:15]] > g] for r in g[::n] if [r] * ~i > g] * -i or p(g, n + 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

Inputs have width and height 21-28. The output size is 3x3 or 4x4. The megasprite is generated from a Conway-style binary sprite with diagonal connectivity, scaled by factor 3-5 for 3x3 sprites and 3-4 for 4x4 sprites. It is placed near the top with row 1 and a random left column. The color box has the same 3x3/4x4 size and is placed near the bottom/right, with a guaranteed one-cell clearance from the large sprite. All color values are nonzero random ARC colors; the small box may contain repeated colors.

## Reference Notes

The ARC-DSL solver selects the largest object as the megasprite and the smallest object as the color box, downscales the megasprite by the ratio of their bounding-box widths, then zeros the color-box cells where the downscaled megasprite is zero. The Code Golf 2025 solution implements the same idea by trying step sizes and sampling every nth row/column against the bottom color box. There is no apparent disagreement between sources.
