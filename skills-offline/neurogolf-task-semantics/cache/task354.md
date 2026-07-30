# task354 Semantics

## Sources

- Current champion builder: `solutions_py/task354.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task354.json`
- ARC-GEN task id: `ddf7fa4f`
- ARC-DSL task id: `ddf7fa4f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ddf7fa4f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ddf7fa4f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task354.py`

## Pattern

The input is a `10x10` scene with three single-cell colored lights on the top row and three gray (`5`) rectangles below them. Each gray rectangle lies horizontally under exactly one light column. The output preserves the scene shape and recolors every gray rectangle with the color of the light above it. Background remains black, and the top-row lights remain their original colors.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    lights = [(c, grid[0][c]) for c in range(w) if grid[0][c] not in (0, 5)]

    seen = [[False] * w for _ in range(h)]
    for r0 in range(1, h):
        for c0 in range(w):
            if seen[r0][c0] or grid[r0][c0] != 5:
                continue
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 5:
                        seen[nr][nc] = True
                        stack.append((nr, nc))

            cols = {c for _, c in cells}
            color = next(light_color for light_col, light_color in lights if light_col in cols)
            for r, c in cells:
                out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN uses a square grid, default size `10`. It chooses three non-gray light colors, places one light in each of three column bands (`0..2`, `4..5`, `7..9`), and creates three gray rectangles. Rectangle widths are `2..5`, heights are `2..7`, rows are at least `2`, and each rectangle is horizontally under only its own light. Rectangles are rejected if they overlap each other or if any rectangle is under the wrong light. The three rectangle top rows are not all equal.

The task is therefore fixed-size and has exactly three light/rectangle pairs in generated data. Black background and gray rectangles are special colors; the light/recolor colors are never gray.

## Reference Notes

ARC-DSL finds all objects, separates singleton objects (the lights) from gray objects, pairs each light with each gray object, filters by vertical matching, then recolors the gray object with the light color. This confirms that the matching predicate is column overlap/alignment, not nearest distance.

The Code Golf 2025 solution repeatedly propagates top-row colors downward through gray cells, effectively letting each gray rectangle inherit the color of the light in its column support. It relies on the fixed `10x10` layout.
