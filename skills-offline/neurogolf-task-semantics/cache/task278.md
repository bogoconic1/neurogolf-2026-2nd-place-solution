# task278 Semantics

## Sources

- Current champion builder: `solutions_py/task278.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task278.json`
- ARC-GEN task id: `b27ca6d3`
- ARC-DSL task id: `b27ca6d3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b27ca6d3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b27ca6d3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task278.py`

## Pattern

Inputs are black-background grids with red pixels. Most red pixels are isolated noise and must remain red. Some red pixels occur as one horizontal or vertical adjacent pair; in the hidden generator these are the two red centers of an "olive" whose surrounding 3x3 neighborhoods were green before the input view mapped green to black. The output restores green (`3`) on the one-cell outbox around each connected red size-2 component, leaving the two red center cells red (`2`) and preserving all other black/red cells.

Equivalently: find each red object of size exactly two under object connectivity, draw the bounding rectangle expanded by one cell in every direction, color that expanded box green, then put the original red cells back on top. Isolated red singleton objects are unchanged.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    seen = [[False] * w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] != 2:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 2:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(comp) != 2:
                continue
            rs = [x for x, _ in comp]
            cs = [y for _, y in comp]
            for rr in range(max(0, min(rs) - 1), min(h, max(rs) + 2)):
                for cc in range(max(0, min(cs) - 1), min(w, max(cs) + 2)):
                    out[rr][cc] = 3
            for rr, cc in comp:
                out[rr][cc] = 2
    return out
```

## Generator Constraints

Random generated grids have width and height in `15..18`. The generator first places sparse isolated red pixels (`color 2`) and removes neighboring red pixels to keep this static noise separated. It then samples horizontal or vertical adjacent red pairs. Around each center in such a pair it draws a 3x3 green (`color 3`) neighborhood, then overwrites the two centers red. The public input maps green back to black (`0`), while the output keeps the original green. The generator avoids overlap between green olive regions, but isolated red noise may appear near borders. ARC-AGI examples include the same size range and border cases.

## Reference Notes

ARC-DSL implements exactly `objects(I, T, F, T)`, `sizefilter(..., TWO)`, `mapply(outbox, ...)`, then `fill(I, THREE, outbox_pixels)`. This confirms the size-2 red object is the semantic trigger and `outbox` is the green region. The Code Golf solution is a recursive compact expression that repeatedly uses neighboring values to propagate the same restoration rule.
