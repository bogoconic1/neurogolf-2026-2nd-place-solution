# task348 Semantics

## Sources

- Current champion builder: `solutions_py/task348.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task348.json`
- ARC-GEN task id: `db3e9e38`
- ARC-DSL task id: `db3e9e38`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_db3e9e38.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_db3e9e38.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task348.py`

## Pattern

The input has a single vertical orange (`7`) line starting at the top edge in
one interior column. The output expands that line into a top-aligned symmetric
wedge/triangle. A cell at row `r`, column `c` is filled if its horizontal
distance from the original line column is small enough: `r < length - abs(c-col)`.
Filled cells with the same column parity as the original line stay orange (`7`),
and filled cells of opposite parity become cyan (`8`). Everything outside the
wedge remains black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    col = next(c for c in range(w) if grid[0][c] == 7)
    length = 0
    while length < h and grid[length][col] == 7:
        length += 1

    out = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        dist = abs(c - col)
        fill_h = length - dist
        if fill_h <= 0:
            continue
        color = 7 if (c - col) % 2 == 0 else 8
        for r in range(fill_h):
            out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN chooses width and height independently from `5..10`. The orange line
column is interior, `2 <= col <= width-3`, so the source column is never too
close to a side edge. The line length is `3..height-1`, always starts at row 0,
and is the only non-black input content. The output wedge can clip horizontally
at the grid boundaries if the line length exceeds distance to a side. The
NeuroGolf output is still the dense padded `[1,10,30,30]` tensor.

## Reference Notes

The ARC-DSL solver finds the orange cells, takes the lower-right corner of that line, shoots diagonals upward-left and upward-right, then shoots upward from those diagonal seeds. It fills the resulting wedge cyan, then filters cells by column parity relative to the line and refills those orange. The Code Golf 2025 solution encodes the same top-down recurrence over columns/rows.
