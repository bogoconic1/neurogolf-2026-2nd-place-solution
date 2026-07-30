# task025 Semantics

## Sources

- Current champion builder: `solutions_py/task025.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task025.json`
- ARC-GEN task id: `1a07d186`
- ARC-DSL task id: `1a07d186`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1a07d186.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1a07d186.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task025.py`

## Pattern

The input contains one or more full straight colored lines and sparse isolated colored pixels on a zero background. In the untransposed generator the lines are vertical; a random transpose branch turns the entire task into the horizontal-line version. The output keeps the full lines, removes all isolated input pixels, and redraws only the isolated pixels whose color matches one of the line colors. Each matching pixel is moved to the cell immediately adjacent to its same-colored line on the side where the pixel came from.

For a vertical line of color `k` at column `L`, an isolated pixel of color `k` at `(r, c)` outputs at `(r, L - 1)` if `c < L`, otherwise `(r, L + 1)`. For a horizontal line of color `k` at row `L`, an isolated pixel of color `k` at `(r, c)` outputs at `(L - 1, c)` if `r < L`, otherwise `(L + 1, c)`. Isolated pixels with colors not used by any line are ignored.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    lines = []

    for c in range(w):
        vals = [grid[r][c] for r in range(h)]
        nz = [v for v in vals if v]
        if nz and len(nz) == h and len(set(nz)) == 1:
            color = nz[0]
            lines.append(("v", c, color))
            for r in range(h):
                out[r][c] = color

    for r in range(h):
        vals = grid[r]
        nz = [v for v in vals if v]
        if nz and len(nz) == w and len(set(nz)) == 1:
            color = nz[0]
            lines.append(("h", r, color))
            for c in range(w):
                out[r][c] = color

    line_by_color = {color: (axis, pos) for axis, pos, color in lines}
    line_cells = set()
    for axis, pos, color in lines:
        if axis == "v":
            line_cells.update((r, pos) for r in range(h))
        else:
            line_cells.update((pos, c) for c in range(w))

    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if not color or (r, c) in line_cells or color not in line_by_color:
                continue
            axis, pos = line_by_color[color]
            if axis == "v":
                out[r][pos - 1 if c < pos else pos + 1] = color
            else:
                out[pos - 1 if r < pos else pos + 1][c] = color
    return out
```

## Generator Constraints

- Active width and height are sampled from 12 through 30.
- In the untransposed form, line columns start at a random column 3 through 6 and then advance by gaps of 6 through 9, so there may be several well-spaced vertical lines.
- Line colors are distinct random colors, plus one extra distinct color used only for irrelevant isolated pixels; the extra color is removed from the line color list before drawing lines.
- Sparse pixels are sampled with probability 0.02, then filtered: no pixel lies on a line column, adjacent to a line column, or on the same row as another pixel of the same color.
- A random transpose branch swaps rows/columns for both input and output, producing the horizontal-line version with the same movement rule.

## Reference Notes

- ARC-DSL identifies singleton objects and non-singleton line objects, filters singleton pixels whose color is among line colors, finds the same-colored line, computes the gravitation vector to that line, shifts the singleton to the adjacent side cell, covers the original singleton pixels, and paints the shifted pixels over the preserved lines.
- The Code Golf 2025 solution repeatedly scans/transposes the grid and uses row/column state to move matching singleton colors toward their line while discarding unmatched pixels.
- The line colors are unique, so a matching isolated pixel has exactly one target line.
