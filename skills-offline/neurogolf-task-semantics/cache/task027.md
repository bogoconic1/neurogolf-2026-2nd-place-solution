# task027 Semantics

## Sources

- Current champion builder: `solutions_py/task027.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task027.json`
- ARC-GEN task id: `1b60fb0c`
- ARC-DSL task id: `1b60fb0c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1b60fb0c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1b60fb0c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task027.py`

## Pattern

The input is a fixed 10x10 grid containing three blue (`1`) rotated copies of the same small connected shape around the center. The missing fourth rotated copy must be drawn in red (`2`) while preserving all existing blue cells. There is a binary offset branch: some generated examples shift the rotated copies by one cell, so the missing red copy uses one of two nearby rotation alignments.

In generator coordinates, each original creature pixel `(r, c)` creates blue cells at three rotated/translated positions:

- `(5 - r + offset, 5 + c)`
- `(4 - c + offset, 5 - r + offset)`
- `(5 + c, 4 + r)`

The output preserves those blue cells and adds red at the fourth position:

- `(4 + r, 4 - c + offset)`

## Readable Python Solver

```python
def solve(grid):
    n = 10
    # The Code Golf solution selects the offset by comparing the center column
    # with the reversed center row. This distinguishes the two generator branches.
    col5 = [grid[r][5] for r in range(n)]
    row5_rev = list(reversed(grid[5]))
    offset = 1 if col5 < row5_rev else 0

    out = [row[:] for row in grid]
    for i in range(n):
        for j in range(n):
            # Rotate/copy the existing blue source into the missing quadrant.
            src_r = (offset - j - 1) % n
            if grid[src_r][i] and out[i][j] == 0:
                out[i][j] = 2
    return out
```

This is equivalent to the compact Code Golf expression `g[i][j] or 2*g[~j+offset][i]` with Python negative indexing.

## Generator Constraints

- Active grid size is always `10x10`.
- The source creature is a connected set sampled by `continuous_creature(randint(6, 12), 4, 4)`.
- Source rows and columns can include small negative/positive coordinates; generated placements stay within the 10x10 canvas after the fixed rotations/translations.
- `offset` is either `0` or `1` and shifts two of the blue copies plus the missing red copy by one cell.
- Input cells are only black (`0`) and blue (`1`). Output cells are black, blue, and red (`2`); red is only painted where the rotated source lands on black.

## Reference Notes

- ARC-DSL rotates the input, collects blue cells in the original and rotated grids, tests a small neighborhood of candidate shifts, chooses the shift with maximum overlap against the original blue cells, and underfills red at the shifted rotated cells.
- Code Golf 2025 shows the offset can be derived from a lexicographic comparison of center column 5 and reversed center row 5, then the missing copy is just a rotated indexed lookup.
- The input blue cells must be preserved; red is an underfill, so it does not overwrite blue.
