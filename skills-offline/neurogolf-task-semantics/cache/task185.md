# task185 Semantics

## Sources

- Current champion builder: `solutions_py/task185.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task185.json`
- ARC-GEN task id: `7837ac64`
- ARC-DSL task id: `7837ac64`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7837ac64.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7837ac64.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task185.py`

## Pattern

The input is a square line grid. Most nonzero cells are grid lines in one line
color. Hidden inside the grid is a `4x4` patch of line intersections whose cells
may be recolored with two or three non-line colors; unmarked intersections keep
the line color and represent black in the decoded patch. The marked patch is
aligned to the line grid and corresponds to a `3x3` output. For each output cell,
look at the corresponding `2x2` block of adjacent intersections in the decoded
`4x4` patch. If all four intersections are the same nonzero, non-line color,
output that color. Otherwise output black (`0`).

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])

    counts = {}
    for row in grid:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0) + 1
    line = max(counts, key=counts.get)

    marks = []
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v != 0 and v != line:
                marks.append((r, c))

    top = min(r for r, _ in marks)
    bottom = max(r for r, _ in marks)
    left = min(c for _, c in marks)
    right = max(c for _, c in marks)
    row_step = (bottom - top) // 3
    col_step = (right - left) // 3
    rows = [top + i * row_step for i in range(4)]
    cols = [left + i * col_step for i in range(4)]

    patch = []
    for r in rows:
        prow = []
        for c in cols:
            v = grid[r][c]
            prow.append(0 if v == line else v)
        patch.append(prow)

    out = [[0 for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            block = [patch[r][c], patch[r][c + 1], patch[r + 1][c], patch[r + 1][c + 1]]
            if block[0] != 0 and all(v == block[0] for v in block):
                out[r][c] = block[0]
    return out
```

## Generator Constraints

- The logical line-grid size is `10` with spacing `2`, `7` with spacing `3`, or
  `6` with spacing `4`; generated ARC input dimensions are therefore `29x29` or
  `27x27` before NeuroGolf padding to `[1,10,30,30]`.
- The line color is a random nonzero ARC color. Marker colors are sampled from
  two colors most of the time and three colors in a smaller branch, excluding
  the line color.
- The hidden decoded pattern is a `3x3` color list containing black plus all
  sampled marker colors. Nonzero entries must touch every margin of the `3x3`
  pattern, so the colored intersection support has no empty outer row or column.
- Adjacent nonzero output cells with different colors are disallowed because
  their generated `2x2` intersection blocks would overlap inconsistently.
- Every sampled marker color must be diagonally connected in the `3x3` pattern.
- The hidden patch top-left logical cell `(brow,bcol)` is chosen so the full
  `4x4` intersection patch fits inside the line grid.

## Reference Notes

ARC-DSL removes the largest foreground object as the grid-line object, crops the remaining colored intersection support, fills selected black objects from their outbox/corner colors, then repeatedly samples rows and columns at the line-grid spacing and downscales the `4x4` intersection patch to `3x3`.

The Code Golf solution is terse but follows the same idea: identify the repeated line-grid background, isolate non-line colors, and collapse the intersection patch to the small output.
