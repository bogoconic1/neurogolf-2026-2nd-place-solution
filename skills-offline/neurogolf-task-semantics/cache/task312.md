# task312 Semantics

## Sources

- Current champion builder: `solutions_py/task312.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task312.json`
- ARC-GEN task id: `c9f8e694`
- ARC-DSL task id: `c9f8e694`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c9f8e694.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c9f8e694.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task312.py`

## Pattern

The visible task grid is a fixed `12 x 12` square. Column 0 contains a row
pattern of colors, sometimes starting with a black row. Several non-overlapping
gray rectangles are placed away from column 0. The output keeps black cells
black, keeps the column-0 pattern, and recolors every gray rectangle cell to the
color stored in column 0 of the same row.

Equivalently: for each row `r`, copy `grid[r][0]` across every nonzero/nonblack
cell in that row; cells that are black in the input remain black.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    for r, row in enumerate(grid):
        label = row[0]
        for c, x in enumerate(row):
            if x != 0:
                out[r][c] = label
            else:
                out[r][c] = 0
    return out
```

## Generator Constraints

- The generated grid size is fixed at `12 x 12`.
- There are 3 or 4 non-overlapping gray rectangles.
- Rectangle widths are `2..6`; heights are `3..7`.
- Rectangle columns are at least 2 in random generation, so column 0 remains the
  row-label/pattern column.
- Rectangle rows may start at row 0 or 1 depending on the random `start` value.
- The row pattern length is 12. If `start == 1`, row 0 is black and the remaining
  rows draw from a random palette of 2 or 3 non-gray colors.
- Gray (`5`) is excluded from the row-label palette.
- The output writes each gray rectangle cell with its row label from column 0.
  Black cells stay black.

## Reference Notes

ARC-GEN gives the exact fixed-size row-label fill rule. ARC-DSL expresses the same rule by cropping the first column, horizontally upscaling it to the full width, and then filling black cells from the original input back to zero. The Code Golf solution uses a compact arithmetic trick equivalent to `x != 0 ? row[0] : 0` for each cell.

No disagreement was found between the references. The NeuroGolf tensor contract still uses the full dense `[1, 10, 30, 30]` one-hot canvas, but the semantic task content is the top-left `12 x 12` region.
