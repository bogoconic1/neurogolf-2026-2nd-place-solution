# task004 Semantics

## Sources

- Current champion builder: `solutions_py/task004.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task004.json`
- ARC-GEN task id: `025d127b`
- ARC-DSL task id: `025d127b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_025d127b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_025d127b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task004.py`

## Pattern

The input is an 8..16 by 8..16 black grid containing one or more colored slanted-outline boxes. Each box is drawn with a top/left 4-connected component and a bottom/right 4-connected component of the same color. The output has the same size. For every color, keep the rightmost 4-connected component fixed and move every other cell of that color one column to the right. In the generator this shifts the top horizontal edge and the left diagonal edge right by one, while the bottom horizontal edge and right diagonal edge stay fixed. Black stays black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    comps_by_color = {}
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 0 or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps_by_color.setdefault(color, []).append(comp)

    out = [row[:] for row in grid]
    for color, comps in comps_by_color.items():
        keep = max(comps, key=lambda comp: max(c for _, c in comp))
        keep_id = id(keep)
        moving = []
        for comp in comps:
            if id(comp) == keep_id:
                continue
            moving.extend(comp)
        for r, c in moving:
            out[r][c] = 0
        for r, c in moving:
            if c + 1 < w:
                out[r][c + 1] = color
    return out
```

## Generator Constraints

ARC-GEN chooses grid width and height from 8..16. It stacks one or more colored outline boxes vertically, starting at row 1 and leaving one blank row between boxes. For each box, `wide` is 4..width-1 and `tall` is 3..wide-1; the column is chosen so the full box fits. Colors are random nonzero ARC colors. The generated shift is always in-bounds: moved top/left cells have a valid column to their right, and the right/bottom component remains fixed. Multiple boxes can appear, but they are vertically separated by blank rows.

## Reference Notes

The ARC-DSL solver uses `objects(I, T, F, T)`, so it finds non-background, single-color, 4-connected components. It groups those components by color, selects the component with greatest `rightmost` coordinate for each color, subtracts that rightmost component from the union of colored cells, then moves the remaining cells `RIGHT`. The Code Golf solution expresses the same row-local right shift by comparing adjacent rows.
