# task148 Semantics

## Sources

- Current champion builder: `solutions_py/task148.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task148.json`
- ARC-GEN task id: `673ef223`
- ARC-DSL task id: `673ef223`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_673ef223.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_673ef223.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task148.py`

## Pattern

The input is a black grid containing two equal-length red (`2`) vertical border segments on opposite left/right edges. The upper red segment is the source portal and the lower red segment is the destination portal. One or more cyan (`8`) marker cells appear on rows aligned with the upper portal, away from the border. The output preserves the red border segments, changes each input cyan marker to yellow (`4`), draws cyan cells from the marker back toward the upper portal edge on that same row, and draws a full cyan horizontal frontier on the corresponding row of the lower portal with the same vertical offset. Black cells not on those rays/frontiers remain black.

If the upper portal is on the left edge, each marker at `(r, c)` becomes yellow, cells `(r, 1..c-1)` become cyan, and the lower row `r + offset` is filled cyan from column `0` through `width-2` while preserving the red cell at the right edge. If the upper portal is on the right edge, this is mirrored: cells between the marker and the right-edge red portal become cyan, and the lower row is filled cyan from column `1` through `width-1` while preserving the left-edge red cell.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]

    left_rows = [r for r in range(h) if grid[r][0] == 2]
    right_rows = [r for r in range(h) if grid[r][w - 1] == 2]
    if min(left_rows) < min(right_rows):
        top_side = "left"
        top_rows = left_rows
        bottom_rows = right_rows
    else:
        top_side = "right"
        top_rows = right_rows
        bottom_rows = left_rows
    offset = min(bottom_rows) - min(top_rows)

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 8:
                continue
            out[r][c] = 4
            rr = r + offset
            if top_side == "left":
                for cc in range(1, c):
                    if out[r][cc] == 0:
                        out[r][cc] = 8
                for cc in range(0, w - 1):
                    if out[rr][cc] == 0:
                        out[rr][cc] = 8
            else:
                for cc in range(c + 1, w - 1):
                    if out[r][cc] == 0:
                        out[r][cc] = 8
                for cc in range(1, w):
                    if out[rr][cc] == 0:
                        out[rr][cc] = 8
    return out
```

## Generator Constraints

ARC-GEN samples width from 8 through 12 and height from 16 through 24. The red portal length is 4 through 6. Before optional horizontal flip, the upper/source portal is a red vertical segment on the left edge and the lower/destination portal is a red vertical segment on the right edge. The source starts near the top, the destination starts near the bottom, and the generator rejects cases where a row would contain red cells on both left and right borders. Horizontal flip mirrors the entire input/output, so the source portal may appear on the right and the destination on the left.

For each offset row within the portal length, the generator independently chooses whether to place a cyan marker on the source row. Marker columns are from `2` through `width-2`. The generated case is accepted only when there are at least two markers. Inputs use only colors `0`, `2`, and `8`; outputs additionally use yellow `4` at the original marker cells and cyan `8` ray/frontier cells.

## Reference Notes

ARC-DSL identifies all cyan marker cells, replaces cyan with yellow, finds the vertical offset between the two red objects, shoots rays from each marker toward the source portal side, shifts the marker cells by the portal offset, and underfills the horizontal frontiers through those shifted cells with cyan. The direction is selected from the side of the uppermost red object: `LEFT` when it is on the left edge, otherwise `RIGHT`.

The Code Golf solution uses row scans and accumulated state to infer the portal side, offset, and marker rows compactly. It agrees with the generator and DSL that the output is same-sized, horizontally mirrored when the input is mirrored, and that fill operations should preserve existing red and yellow cells.
