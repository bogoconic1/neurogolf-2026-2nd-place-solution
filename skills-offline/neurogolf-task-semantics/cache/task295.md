# task295 Semantics

## Sources

- Current champion builder: `solutions_py/task295.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task295.json`
- ARC-GEN task id: `bbc9ae5d`
- ARC-DSL task id: `bbc9ae5d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_bbc9ae5d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_bbc9ae5d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task295.py`

## Pattern

The input is a single row of even width `W`, where `W` is in `6, 8, ..., 18`.
A run of length `L` starts at column 0 and uses one non-background color. The
remaining cells are background.

The output has height `W // 2` and width `W`. Row `r` contains the same color
from columns `0` through `L + r - 1`, then background to the right. The colored
run therefore grows by one cell on every output row, forming a left-aligned
staircase or triangular wedge.

## Readable Python Solver

```python
def solve(grid):
    row = grid[0]
    width = len(row)
    color = next((v for v in row if v != 0), 0)
    length = 0
    while length < width and row[length] == color:
        length += 1

    out = []
    for r in range(width // 2):
        run = min(width, length + r)
        out.append([color if c < run else 0 for c in range(width)])
    return out
```

## Generator Constraints

ARC-GEN samples choose `width = 2 * randint(3, 9)`, so valid widths are even
from 6 through 18. `length` is in `1..width//2 + 1`, and the foreground color is
non-background. The input grid is exactly one row. The output width is the same
as the input width and output height is `width // 2`.

Important edge cases:

- the foreground run always starts at column 0
- foreground color is nonzero
- output rows never exceed width because `length <= width//2 + 1` and there are
  `width//2` rows
- outside the active output height/width must remain empty in the NeuroGolf
  30x30 tensor, not background channel 0

## Reference Notes

ARC-DSL computes the input width, halves it to get the output height, vertically upscales the input row by that height, then shoots rays to the right from the foreground cells to fill the growing staircase. The Code Golf solution starts with the input row and repeatedly prepends the first cell while dropping the last cell, which shifts the run boundary one step to the right each row.
