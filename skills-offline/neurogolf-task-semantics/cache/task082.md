# task082 Semantics

## Sources

- Current champion builder: `solutions_py/task082.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task082.json`
- ARC-GEN task id: `3ac3eb23`
- ARC-DSL task id: `3ac3eb23`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3ac3eb23.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3ac3eb23.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task082.py`

## Pattern

The input is a 6-row grid with variable width. Only row 0 contains colored markers; all lower rows are black. For each marker color at column `c`, the output repeats a two-row stripe pattern three times: even output rows keep that color at column `c`, and odd output rows place that color at the horizontal neighbors `c-1` and `c+1`. All other cells are black.

Equivalently, output row 0 is the input top row, output row 1 is the horizontal dilation of that row to immediate left/right neighbors with marker columns cleared, and rows 2/3 and 4/5 repeat rows 0/1.

There is no tie-breaking. Marker spacing prevents neighbor colors from colliding.

## Readable Python Solver

```python
def solve(grid):
    top = grid[0]
    width = len(top)
    neighbor = [0] * width
    for c, color in enumerate(top):
        if color:
            if c - 1 >= 0:
                neighbor[c - 1] = color
            if c + 1 < width:
                neighbor[c + 1] = color
    return [top[:], neighbor, top[:], neighbor[:], top[:], neighbor[:]]
```

## Generator Constraints

ARC-GEN samples width from 5..15 and fixed height 6. The first marker column is 1 or 2. Subsequent markers are spaced by 3 or 4 columns while `col + 1 < width`, so every marker has both horizontal neighbors in bounds and neighbor writes do not collide. Marker colors are random non-background ARC colors. The input grid contains the markers only in row 0; rows 1..5 are black.

The output is the same 6-row height and sampled width. Even rows contain the original markers. Odd rows contain only each marker color shifted one column left and one column right.

## Reference Notes

ARC-DSL finds the foreground objects, recolors the immediate neighbors of each marker with that marker color, paints those neighbors, then takes the first third of the grid and repeats it vertically to make six rows.

The Code Golf 2025 solution computes `r = g[0]` and `[max(left, right)]` for the neighbor row, then returns that two-row block repeated three times. The use of `max` is safe because generator spacing prevents overlapping neighbor colors.
