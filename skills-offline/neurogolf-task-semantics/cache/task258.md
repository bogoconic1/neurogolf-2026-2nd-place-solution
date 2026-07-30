# task258 Semantics

## Sources

- Current champion builder: `solutions_py/task258.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task258.json`
- ARC-GEN task id: `a699fb00`
- ARC-DSL task id: `a699fb00`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a699fb00.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a699fb00.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task258.py`

## Pattern

The input and output have the same square size. The background is black (`0`). Some rows contain odd-length horizontal runs whose visible pattern is blue (`1`) at every even position in the run, with black gaps between them in the input. The output fills each black gap that has a blue cell immediately to its left and right with red (`2`). Existing blue cells stay blue and all other black cells stay black.

Equivalently, for every cell `(r, c)`, output red at `(r, c)` exactly when `grid[r][c] == 0`, `grid[r][c-1] == 1`, and `grid[r][c+1] == 1`. All other cells keep their input color.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(1, w - 1):
            if grid[r][c] == 0 and grid[r][c - 1] == 1 and grid[r][c + 1] == 1:
                out[r][c] = 2
    return out
```

## Generator Constraints

The grid is square with size `5..10`. The generator creates one or more horizontal segments on distinct rows. Each segment has odd length `2*k + 1`, where `k` ranges from `1` to `(size // 2) - 1`, so lengths are at least 3 and never exceed the row width. A segment start column is chosen so the whole segment fits inside the grid.

Rows are selected by a random walk: the first row is `0` or `1`; after each segment, the next row advances by `1`, `2`, or `3`, with step 2 most common. There can therefore be adjacent active rows, rows separated by gaps, and active rows at the first or last grid row. Only colors `0`, `1`, and `2` appear. The input contains black background plus blue cells at alternating positions inside each generated segment. The output is identical except the between-blue positions of those segments are red.

## Reference Notes

The ARC-DSL solver computes all blue cells, shifts that set left and right, intersects the shifted sets, then fills the intersection with red. This confirms that red is placed only at cells horizontally between two blue neighbors and does not depend on row grouping, segment length, or object counting.

The Code Golf 2025 solution performs a string/list substitution replacing the pattern `1, 0, 1` with `1, 2, 1`, which matches the local three-cell rule. There is no ambiguity between train/test/reference implementations.
