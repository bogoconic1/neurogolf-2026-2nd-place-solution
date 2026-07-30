# task120 Semantics

## Sources

- Current champion builder: `solutions_py/task120.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task120.json`
- ARC-GEN task id: `50cb2852`
- ARC-DSL task id: `50cb2852`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_50cb2852.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_50cb2852.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task120.py`

## Pattern

The input is a rectangular grid containing 2 to 4 non-overlapping solid colored rectangles on a black background. Rectangle colors are from 1, 2, and 3, and at least two distinct colors are present in each generated task instance. The output preserves the grid size, black background, and colored rectangle borders. For every rectangle, all strict interior cells become cyan (`8`), while the outer border cells keep the rectangle's original color.

Equivalently: detect each connected nonzero rectangular object, compute its bounding box, and recolor cells whose row is strictly between the top and bottom and whose column is strictly between the left and right to cyan.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    seen = [[False] * w for _ in range(h)]

    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or grid[sr][sc] == 0:
                continue
            color = grid[sr][sc]
            stack = [(sr, sc)]
            seen[sr][sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))

            r0 = min(r for r, _ in cells)
            r1 = max(r for r, _ in cells)
            c0 = min(c for _, c in cells)
            c1 = max(c for _, c in cells)
            for r in range(r0 + 1, r1):
                for c in range(c0 + 1, c1):
                    out[r][c] = 8
    return out
```

## Generator Constraints

ARC-GEN chooses width in `[12, 14]`; height is `width + randint(-2, 2)`, so generated height is roughly 10 to 16. There are 2 to 4 rectangles. Each rectangle has width and height in `[3, 8]`, with top-left positions selected so the rectangle stays within the grid. Rectangles are required not to overlap and to have at least a one-cell separation margin (`common.overlaps(..., 1)` must be false). Colors are independently chosen from `[1, 3]` until at least two different colors occur. Inputs contain solid rectangles; outputs replace only strict interiors with cyan.

The validate examples include sizes outside the random width range (`width=15` and `width=11`), so a solver should not hard-code the 12 to 14 generated width range. The NeuroGolf tensor is still the standard `[1,10,30,30]` one-hot canvas.

## Reference Notes

The ARC-DSL solver computes `objects(I, T, F, T)`, maps `inbox` and `backdrop` over each object, and fills those interior backdrops with color `EIGHT`. That confirms the rule is object-interior filling rather than color-specific behavior.

The Code Golf 2025 one-liner checks a cell against nearby equal-color neighbors and emits `8` for cells that are surrounded as rectangle interiors. It is a compact local-neighborhood view of the same rule.
