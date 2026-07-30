# task100 Semantics

## Sources

- Current champion builder: `solutions_py/task100.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task100.json`
- ARC-GEN task id: `445eab21`
- ARC-DSL task id: `445eab21`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_445eab21.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_445eab21.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task100.py`

## Pattern

The input is a 10x10 black grid containing two non-overlapping rectangular
outline boxes in two nonzero colors. Choose the box with the larger area
`height * width` and return a 2x2 output canvas filled with that box color.
The generator rejects equal areas, so there is no tie in normal data. The
input may be transposed after drawing; this swaps height/width but preserves
area and color.

## Readable Python Solver

```python
def solve(grid):
    colors = sorted({v for row in grid for v in row if v != 0})
    best_color = None
    best_area = -1

    for color in colors:
        cells = [(r, c) for r, row in enumerate(grid)
                 for c, value in enumerate(row) if value == color]
        if not cells:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        height = max(rs) - min(rs) + 1
        width = max(cs) - min(cs) + 1
        area = height * width
        if area > best_area:
            best_area = area
            best_color = color

    return [[best_color, best_color], [best_color, best_color]]
```

## Generator Constraints

- Input size is always 10x10; output size is always 2x2.
- Exactly two rectangular outline boxes are drawn on a black background.
- Box heights are generated as `tall0 in [3, 7]` and
  `tall1 in [3, 10 - tall0]`; widths are each in `[3, 10]`.
- The two box areas are explicitly regenerated until unequal.
- The first box starts at row 0. The second starts immediately below it, with
  one optional blank row when the two heights do not exactly fill the grid.
- Each box has an independently random horizontal offset that keeps it in the
  10-column grid.
- The two colors are distinct nonzero colors from `common.random_colors(2)`.
- With 50% probability the whole input is transposed after drawing.

## Reference Notes

- ARC-DSL computes connected/color objects, selects the object with maximal `height * width`, reads its color, and returns a 2x2 canvas of that color.
- ARC-GEN clarifies that equal-area ties are not produced and that transpose is a legal variant.
- The Code Golf 2025 solution uses row and column color counts to infer rectangle area for each nonzero color, then returns the color with the largest product. This matches the outline-box semantics because every rectangle has full colored top/bottom rows and left/right columns.
