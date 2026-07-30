# task019 Semantics

## Sources

- Current champion builder: `solutions_py/task019.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task019.json`
- ARC-GEN task id: `10fcaaa3`
- ARC-DSL task id: `10fcaaa3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_10fcaaa3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_10fcaaa3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task019.py`

## Pattern

The input is a small `height x width` grid, with `height,width` from 2 through 6. It contains one foreground color other than cyan. The foreground cells occur in a few rows separated by gaps.

The output size is exactly `2*height x 2*width`. First tile the input into all four quadrants, preserving the foreground color. Then, for each tiled foreground cell, fill cyan (`8`) into its diagonal neighbor cells when those cells are inside the output and still background. Foreground cells take precedence over cyan if a diagonal neighbor would overlap an actual foreground copy.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(2 * w)] for _ in range(2 * h)]
    cells = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                cells.append((r, c, grid[r][c]))

    for r, c, color in cells:
        for dr0, dc0 in ((0, 0), (h, 0), (0, w), (h, w)):
            rr, cc = r + dr0, c + dc0
            for dr, dc in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < 2 * h and 0 <= nc < 2 * w and out[nr][nc] == 0:
                    out[nr][nc] = 8

    for r, c, color in cells:
        for dr0, dc0 in ((0, 0), (h, 0), (0, w), (h, w)):
            out[r + dr0][c + dc0] = color
    return out
```

## Generator Constraints

- Input width and height are each independently 2 through 6.
- Rows containing foreground cells are chosen by starting at row 0 or 1 and advancing by 2 or 3, so active rows are separated.
- Each active row has one foreground cell at a random column.
- The foreground color is any random nonzero color except cyan (`8`).
- Output dimensions are at most 12 by 12.
- Cyan cells are drawn first at diagonal neighbors of every tiled foreground cell; foreground tiled cells are drawn second and may cover cyan.

## Reference Notes

- ARC-DSL computes the least/nonzero foreground color, tiles the input horizontally and vertically, collects the foreground positions in the tiled grid, gets their diagonal/intercardinal neighbors, and underfills those locations with cyan.
- The Code Golf solution is a compact recursive transpose/shift implementation of the same 2x tiling plus diagonal cyan fill.
- There is no color ambiguity: cyan never appears as the source foreground color, and background remains 0 except where underfilled with cyan.
