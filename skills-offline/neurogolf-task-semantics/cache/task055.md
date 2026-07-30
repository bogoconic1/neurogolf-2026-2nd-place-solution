# task055 Semantics

## Sources

- Current champion builder: `solutions_py/task055.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task055.json`
- ARC-GEN task id: `272f95fa`
- ARC-DSL task id: `272f95fa`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_272f95fa.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_272f95fa.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task055.py`

## Pattern

The input is a rectangular grid containing only background `0` and cyan separator lines `8`. The separator lines form a 3-by-3 table with three variable-height row bands and three variable-width column bands. The output keeps the cyan separator lines and the four corner background cells unchanged, then recolors the five cross-shaped table cells with fixed colors:

- top-middle cell -> `2`
- middle-left cell -> `4`
- center cell -> `6`
- middle-right cell -> `3`
- bottom-middle cell -> `1`

The output size is the same as the input size. The rule is geometric: identify the unique non-border zero component as the center cell, then select the zero components vertically or horizontally aligned with it.

## Readable Python Solver

```python
from collections import deque

def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    seen = [[False] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] != 0:
                continue
            q = deque([(r, c)])
            seen[r][c] = True
            cells = []
            while q:
                x, y = q.popleft()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < h and 0 <= ny < w and not seen[nx][ny] and grid[nx][ny] == 0:
                        seen[nx][ny] = True
                        q.append((nx, ny))
            comps.append(cells)

    def bounds(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), max(rs), min(cs), max(cs)

    def borders(cells):
        r0, r1, c0, c1 = bounds(cells)
        return r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1

    center = [cells for cells in comps if not borders(cells)][0]
    cr0, cr1, cc0, cc1 = bounds(center)

    for r, c in center:
        out[r][c] = 6

    for cells in comps:
        if cells is center:
            continue
        r0, r1, c0, c1 = bounds(cells)
        same_cols = c0 == cc0 and c1 == cc1
        same_rows = r0 == cr0 and r1 == cr1
        if same_cols and r1 < cr0:
            color = 2
        elif same_cols and r0 > cr1:
            color = 1
        elif same_rows and c1 < cc0:
            color = 4
        elif same_rows and c0 > cc1:
            color = 3
        else:
            continue
        for r, c in cells:
            out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN samples three row band heights and three column band widths, each from `1..10`. It inserts one-cell cyan separator lines between adjacent bands, so the generated height and width are `sum(rows) + 2` and `sum(cols) + 2`. Validation examples use the same 3-by-3 structure with fixed separator color `8`, background `0`, and output fill colors `(1, 2, 3, 4, 6)`. Bands may be as thin as one cell or as wide/tall as ten cells. The center zero component is the only zero component not touching the outer border.

## Reference Notes

The ARC-DSL solver extracts all zero components, finds the unique non-bordering component as the center, then selects zero objects with matching columns (top-middle and bottom-middle) and matching rows (middle-left and middle-right). It fills center with `6`, top-middle with `2`, bottom-middle with `1`, middle-left with `4`, and middle-right with `3`. The Code Golf solution encodes the same fixed cross-color assignment from the separator geometry in a compact recursive expression. There is no color ambiguity in the references: the output palette is fixed.
