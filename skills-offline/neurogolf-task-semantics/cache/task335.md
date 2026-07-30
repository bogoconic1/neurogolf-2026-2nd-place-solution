# task335 Semantics

## Sources

- Current champion builder: `solutions_py/task335.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task335.json`
- ARC-GEN task id: `d4a91cb9`
- ARC-DSL task id: `d4a91cb9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d4a91cb9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d4a91cb9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task335.py`

## Pattern

The input is a rectangular grid, width and height 10 through 20, with exactly one red `2` marker and one cyan `8` marker on black background. The output keeps both markers and draws a yellow `4` Manhattan L path between them. The bend point is at the red marker row and cyan marker column. Existing marker cells are not overwritten.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    red = cyan = None
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                red = (r, c)
            elif grid[r][c] == 8:
                cyan = (r, c)
    rr, rc = red
    cr, cc = cyan
    for r in range(min(cr, rr), max(cr, rr) + 1):
        if out[r][cc] == 0:
            out[r][cc] = 4
    for c in range(min(cc, rc), max(cc, rc) + 1):
        if out[rr][c] == 0:
            out[rr][c] = 4
    return out
```

## Generator Constraints

- Width and height are each sampled from 10 through 20.
- Red and cyan marker rows are distinct samples from `1..height-3`.
- Red and cyan marker columns are distinct samples from `1..width-3`.
- Only colors 0, 2, 4, and 8 appear in the output.
- The generated output is the HPWL/L-path between markers.

## Reference Notes

ARC-DSL finds the cyan and red cells, forms bend coordinate `(red row, cyan col)`, connects that bend to each marker, combines the segments, and underfills yellow. Code Golf implements the same row/column membership test.
