# task200 Semantics

## Sources

- Current champion builder: `solutions_py/task200.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task200.json`
- ARC-GEN task id: `8403a5d5`
- ARC-DSL task id: `8403a5d5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8403a5d5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8403a5d5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task200.py`

## Pattern

The input is a fixed 10x10 black grid with exactly one non-gray colored pixel on the bottom row. If the source pixel is at column `c` with color `k`, the output draws a rightward zig-zag path. Every even offset column from `c` (`c, c+2, c+4, ...`) is filled from top to bottom with color `k`. Between those color columns, a single gray (`5`) marker is placed at the edge where the path turns: row `0` for columns `c+1, c+5, ...` and row `9` for columns `c+3, c+7, ...`. Columns left of `c` remain black.

Equivalently, starting at `(9, c)`, fill the current column with `k` to the opposite vertical edge, move one column right and put gray at that edge, then move one column right and fill the next column back to the other edge, repeating until the grid's right edge.

## Readable Python Solver

```python
def solve(grid):
    size = 10
    out = [[0 for _ in range(size)] for _ in range(size)]
    c0 = next(c for c, v in enumerate(grid[9]) if v != 0)
    color = grid[9][c0]
    row = size - 1
    col = c0
    while col < size:
        d = -1 if row else 1
        out[row][col] = color
        while 0 <= row + d < size:
            row += d
            out[row][col] = color
        col += 1
        if col >= size:
            break
        out[row][col] = 5
        col += 1
    return out
```

## Generator Constraints

- Grid size is always `10x10`.
- The input has exactly one nonzero pixel at row `9`, column `0..9`.
- The source color is random but excludes gray (`5`).
- Output has the same `10x10` size.
- The original color columns are `c, c+2, c+4, ...`.
- Gray turn markers are singleton cells at alternating top/bottom edges in columns `c+1, c+3, c+5, ...`.

## Reference Notes

The ARC-DSL solver identifies the object color and leftmost/source column, fills every same-parity column from that source to the right with the object color, then paints gray at top-row columns `c+1 mod 4` and bottom-row columns `c+3 mod 4`.

The Code Golf 2025 solution encodes the same pattern by finding the source column in the bottom row and repeatedly emitting row templates that alternate between full color columns and gray edge markers.
