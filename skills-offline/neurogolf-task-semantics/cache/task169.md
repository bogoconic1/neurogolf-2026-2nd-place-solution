# task169 Semantics

## Sources

- Current champion builder: `solutions_py/task169.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task169.json`
- ARC-GEN task id: `6e82a1ae`
- ARC-DSL task id: `6e82a1ae`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6e82a1ae.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6e82a1ae.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task169.py`

## Pattern

The input is a 10x10 black grid containing 4 to 6 separated gray (`5`) sprites. Each sprite is a small connected object with exactly 2, 3, or 4 gray cells. The output preserves the same occupied cells and recolors each whole sprite by its size:

- size 2 object -> color `3`
- size 3 object -> color `2`
- size 4 object -> color `1`

Background remains black. There are no other input colors and no object movement, expansion, cropping, or shape normalization; the only transformation is object-size classification and recoloring.

## Readable Python Solver

```python
from collections import deque


def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    seen = [[False] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] != 5:
                continue
            q = deque([(r, c)])
            seen[r][c] = True
            comp = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 5:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            color = 5 - len(comp)
            for rr, cc in comp:
                out[rr][cc] = color
    return out
```

## Generator Constraints

ARC-GEN uses fixed 10x10 grids. It samples `num_sprites` uniformly from 4 through 6. Each sprite is generated from a small bounding box: widths are usually 2 or 3, heights 1 or 2, with occasional 1x4 or 4x1 `I` sprites. Some 2x2 boxes drop one corner, and 2x3/3x2 boxes drop two outer-corner cells. The generated connected component sizes are therefore always 2, 3, or 4 cells.

Sprite bounding boxes are placed non-overlapping with at least one cell of separation (`common.overlaps(..., 1)` must be false), so connected-component labeling on gray cells is unambiguous. All input sprite cells are gray (`5`); output cells are written as `5 - count`, so only colors 1, 2, and 3 appear on objects.

## Reference Notes

The ARC-DSL solver extracts gray objects, filters them by size 2, 3, and 4, then fills the size-2 cells with 3, size-3 cells with 2, and size-4 cells with 1. The compact Code Golf solution repeatedly transforms/rotates the grid while accumulating object-size evidence, but it implements the same size-to-color mapping.
