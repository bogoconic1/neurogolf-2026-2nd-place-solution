# task273 Semantics

## Sources

- Current champion builder: `solutions_py/task273.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task273.json`
- ARC-GEN task id: `af902bf9`
- ARC-DSL task id: `af902bf9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_af902bf9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_af902bf9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task273.py`

## Pattern

The grid is a 10x10 black canvas containing one or two axis-aligned rectangles marked only by their four yellow (color 4) corners. Each rectangle has width and height between 3 and 6. The output keeps the yellow corners and fills every strict interior cell of each marked rectangle with red (color 2). All other cells remain black. In the two-box branch, the generator places one rectangle in the upper-left partition and one in the lower-right partition after splitting the grid at a row and column in 4..6; an optional vertical flip may be applied to both input and output.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    yellow = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 4]

    for r1, c1 in yellow:
        for r2, c2 in yellow:
            if r2 <= r1 or c2 <= c1:
                continue
            if grid[r1][c2] == 4 and grid[r2][c1] == 4:
                for rr in range(r1 + 1, r2):
                    for cc in range(c1 + 1, c2):
                        out[rr][cc] = 2
    return out
```

## Generator Constraints

- Grid size is always 10x10.
- Colors used by the generator are black 0, red 2, and yellow 4.
- There are one or two rectangles. If there is one rectangle, it may be placed anywhere in the 10x10 grid. If there are two, the grid is split into two diagonal partitions by a row and column chosen in 4..6; one rectangle is placed in each partition.
- Rectangle width and height are each 3..6, bounded by the selected partition.
- The input contains only the four yellow corners of each rectangle; the output adds red to cells with row strictly between top/bottom corners and column strictly between left/right corners.
- A vertical flip may be applied, so optimizations must not assume the first generated rectangle is visually above the second in final coordinates.

## Reference Notes

The ARC-DSL solver finds all pairs of yellow cells connected by a horizontal or vertical line, underfills those line segments with a sentinel, then treats the sentinel-marked rectangular frames as objects. It fills the inbox/backdrop of those frame objects red and replaces the sentinel with black. This confirms that only the rectangle interiors become red and that non-corner edge cells are not yellow in the output.

The Code Golf solution encodes the same parity/interior rule compactly while scanning rows and columns: yellow corners toggle state and cells inside both the vertical and horizontal corner spans become red.
