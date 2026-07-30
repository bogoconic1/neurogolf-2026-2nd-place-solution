# task099 Semantics

## Sources

- Current champion builder: `solutions_py/task099.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task099.json`
- ARC-GEN task id: `444801d8`
- ARC-DSL task id: `444801d8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_444801d8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_444801d8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task099.py`

## Pattern

The 10x10 visible ARC canvas contains one or two fixed-width blue outline boxes.
Each box is five columns wide and has one non-blue sample pixel at the horizontal
center of the box. The answer keeps the original input pixels, and underpaints
the whole box backdrop shifted one row upward with that sample color. Blue
outline pixels and the sample color pixel win over the underpaint because the
operation is an underpaint of the recolored shifted object backdrop.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]

    def blue_component(sr, sc):
        stack = [(sr, sc)]
        seen[sr][sc] = True
        cells = []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc]:
                    if grid[nr][nc] == 1:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
        return cells

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 1 or seen[r][c]:
                continue
            cells = blue_component(r, c)
            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)

            sample = 0
            for rr in range(max(0, r0), min(h, r1 + 1)):
                for cc in range(max(0, c0), min(w, c1 + 1)):
                    if grid[rr][cc] not in (0, 1):
                        sample = grid[rr][cc]

            for rr in range(max(0, r0 - 1), min(h, r1)):
                for cc in range(max(0, c0), min(w, c1 + 1)):
                    if out[rr][cc] == 0:
                        out[rr][cc] = sample
    return out
```

## Generator Constraints

Inputs are generated on a 10x10 square canvas. There is always a top box and
sometimes a second lower box. Box width is always 5. The top box starts in
column 0 or 1 and has height 5, except it can become height 6 when the lower box
has height 4 and the generator chooses that branch. The lower box, when present,
starts at column 4 and row `10 - height`, where height is 4 or 5. If the two
heights leave a blank row, the top box may be shifted down by one row. Each box
gets a distinct random non-blue color, and the color sample is placed at `(top +
height // 2, left + 2)`. The blue outline consists of the vertical sides below
the top row, the row one below the top except the center sample column, and the
bottom row. The output includes the colored rectangle fill, while the input only
contains the blue outline and the center sample pixel.

## Reference Notes

ARC-DSL filters the blue objects, finds the least non-background color adjacent to each object's delta/backdrop, shifts each object's backdrop up, recolors that backdrop with the sample color, and underpaints it into the input. This confirms that original blue/sample pixels take priority over the fill. The Code Golf 2025 solution is very compressed but follows the same recursive column/row propagation idea. There is no disagreement between the references and the generator.
