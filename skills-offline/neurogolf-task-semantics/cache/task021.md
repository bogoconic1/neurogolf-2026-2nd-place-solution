# task021 Semantics

## Sources

- Current champion builder: `solutions_py/task021.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task021.json`
- ARC-GEN task id: `1190e5a7`
- ARC-DSL task id: `1190e5a7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1190e5a7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1190e5a7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task021.py`

## Pattern

The input is a rectangular grid filled mostly with one background color and divided by full-width horizontal separator lines and full-height vertical separator lines in a second color. The separators carve the grid into row blocks and column blocks. The output is a solid rectangle of the background color whose height is the number of row blocks and whose width is the number of column blocks. Equivalently, output height is `horizontal_separator_count + 1`, and output width is `vertical_separator_count + 1`.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter
    h, w = len(grid), len(grid[0])
    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]
    row_seps = sum(1 for row in grid if any(v != bg for v in row) and all(v != bg for v in row))
    col_seps = 0
    for c in range(w):
        col = [grid[r][c] for r in range(h)]
        if any(v != bg for v in col) and all(v != bg for v in col):
            col_seps += 1
    return [[bg for _ in range(col_seps + 1)] for _ in range(row_seps + 1)]
```

## Generator Constraints

- Row block thicknesses and column block thicknesses are generated independently, with 2 through 7 blocks in each direction.
- Each thickness is between 1 and 15, biased toward smaller values by `16 - int(sqrt(randint(1,255)))`.
- The input width is `sum(cols) + len(cols) - 1`; input height is `sum(rows) + len(rows) - 1`, so the grid can be rectangular and up to the NeuroGolf 30x30 bound.
- Two distinct colors are chosen: one background color and one separator color.
- Separator rows and columns are complete full-grid lines in the separator color; output contains only the background color.

## Reference Notes

- ARC-DSL uses `mostcolor` for the background, `frontiers` to find full horizontal/vertical separator lines, separates vertical and horizontal frontiers, takes their counts, increments each count, and returns a background `canvas` of that size.
- Code Golf 2025 recursively strips repeated separator structure; the compact behavior is still counting alternating blocks separated by full lines.
- Empty rows/columns do not occur; all separators are full lines, and the background is the most common color.
