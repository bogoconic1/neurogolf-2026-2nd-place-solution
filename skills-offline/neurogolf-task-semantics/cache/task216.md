# task216 Semantics

## Sources

- Current champion builder: `solutions_py/task216.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task216.json`
- ARC-GEN task id: `8efcae92`
- ARC-DSL task id: `8efcae92`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8efcae92.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8efcae92.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task216.py`

## Pattern

The input is a 20x20 grid containing 3 or 4 separated filled blue rectangles (color 1). Each rectangle contains some red pixels (color 2). The number of red pixels is unique for each rectangle. The output is the full subgrid of the rectangle with the largest red-pixel count, preserving both the blue fill and the red pixels.

The ARC-GEN generator always makes rectangle 0 have the maximum red count, but the rectangle position, width, height, and red-pixel layout vary. A correct solver should not rely on draw order in the visible grid; it should find the object/rectangle with largest red count and crop that rectangle.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = [[False for _ in range(w)] for _ in range(h)]
    best = None

    for r0 in range(h):
        for c0 in range(w):
            if seen[r0][c0] or grid[r0][c0] == 0:
                continue

            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] != 0:
                        seen[nr][nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in cells]
            cols = [c for _, c in cells]
            top, bottom = min(rows), max(rows)
            left, right = min(cols), max(cols)
            red_count = sum(1 for r, c in cells if grid[r][c] == 2)
            candidate = (red_count, top, left, bottom, right)
            if best is None or red_count > best[0]:
                best = candidate

    _, top, left, bottom, right = best
    return [row[left:right + 1] for row in grid[top:bottom + 1]]
```

## Generator Constraints

- Grid size is 20x20.
- There are 3 or 4 rectangles.
- Rectangles are non-overlapping with at least a one-cell separation under `common.overlaps(..., 1)`.
- Rectangle widths are 4..18 and heights are 3..18.
- Each rectangle is fully filled with blue (1), then some cells are overwritten red (2).
- Red pixels are sampled inside each rectangle with probability around 0.2, and every rectangle has at least one red pixel.
- Red counts are unique across rectangles, and the first generated rectangle is constrained to have the largest red count.
- Output size is dynamic and equals the winning rectangle height by width. In NeuroGolf this is represented inside the standard dense `[1, 10, 30, 30]` output tensor with padding outside the actual crop.

## Reference Notes

- ARC-DSL solves this as: find objects, filter by blue-colored rectangles, choose the object with maximum red-pixel count (`size(delta(object))`), and return the object subgrid.
- Code Golf brute-forces possible crop windows and scores them by red count, blue count, and zero count. This reinforces that the target is a source-backed rectangle crop rather than a recoloring task.
- There is no tie ambiguity in generated cases because red counts are unique.
