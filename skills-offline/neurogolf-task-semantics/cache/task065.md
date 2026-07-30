# task065 Semantics

## Sources

- Current champion builder: `solutions_py/task065.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task065.json`
- ARC-GEN task id: `2dc579da`
- ARC-DSL task id: `2dc579da`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_2dc579da.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_2dc579da.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task065.py`

## Pattern

The input is a square grid of size `(2*n + 1) x (2*n + 1)`. The middle row and middle column form a cross in one line color. Every non-cross cell is the background color except for one single odd dot in one of the four quadrants. The output is the `n x n` quadrant panel containing that odd dot: all cells are background except the dot at its local quadrant coordinate. The center cross is not copied to the output.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    n = h // 2
    counts = {}
    for row in grid:
        for value in row:
            counts[value] = counts.get(value, 0) + 1

    dot_color = min(counts, key=counts.get)

    # On generated cases with n >= 2 the background is the most frequent color.
    # For n == 1 the line color can be most frequent, but the 1x1 output is the
    # dot cell, so the background choice is unobserved.
    bg_color = max(counts, key=counts.get)

    dot_r = dot_c = None
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            if value == dot_color:
                dot_r, dot_c = r, c
                break
        if dot_r is not None:
            break

    local_r = dot_r if dot_r < n else dot_r - n - 1
    local_c = dot_c if dot_c < n else dot_c - n - 1

    out = [[bg_color for _ in range(n)] for _ in range(n)]
    out[local_r][local_c] = dot_color
    return out
```

## Generator Constraints

- `n` is sampled from `1..7`, so the raw ARC input is at most `15x15` and the output panel is at most `7x7`.
- The dot row is sampled inside either the top or bottom quadrant, never on the center row. The dot column is sampled inside either the left or right quadrant, never on the center column.
- The line color, dot color, and background color are three distinct digits.
- The generated center cross always spans the full input width and height.
- For `n == 1`, the line color has more cells than the background. This does not affect the visible output because the only output cell is the dot.

## Reference Notes

The ARC-DSL solver splits the grid around the middle row/column into the four quadrants and returns the quadrant with the largest number of distinct colors, which is exactly the quadrant containing the odd dot. The Code Golf solution recursively searches the same quadrant structure and falls back to the rare color when the current panel is already the answer. The ARC-GEN source gives the important size bound (`n <= 7`) and guarantees a single dot off the cross.
