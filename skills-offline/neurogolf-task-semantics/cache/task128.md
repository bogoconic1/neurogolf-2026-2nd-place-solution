# task128 Semantics

## Sources

- Current champion builder: `solutions_py/task128.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task128.json`
- ARC-GEN task id: `5521c0d9`
- ARC-DSL task id: `5521c0d9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5521c0d9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5521c0d9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task128.py`

## Pattern

The input is a `15 x 15` grid containing three solid rectangular boxes resting on the bottom edge. The boxes use colors `1`, `2`, and `4`, one box per color, in shuffled horizontal order. Each box has width `1..5` and height `1..7`; boxes are horizontally disjoint with at least a one-column gap. All other cells are black (`0`).

The output is black except for each box copied straight upward by exactly its own height. A box occupying input rows `15-h .. 14` and columns `c .. c+w-1` becomes the same `h x w` rectangle at output rows `15-2*h .. 14-h` and the same columns. The original bottom copy is removed.

## Readable Python Solver

```python
def solve(grid):
    H, W = len(grid), len(grid[0])
    out = [[0 for _ in range(W)] for _ in range(H)]
    seen = set()
    for c in range(W):
        color = grid[H - 1][c]
        if color == 0 or color in seen:
            continue
        seen.add(color)

        cols = [cc for cc in range(W) if grid[H - 1][cc] == color]
        left, right = min(cols), max(cols)
        h = 0
        while h < H and any(grid[H - 1 - h][cc] == color for cc in range(left, right + 1)):
            h += 1

        for r in range(H - h, H):
            for cc in range(left, right + 1):
                if grid[r][cc] == color:
                    out[r - h][cc] = color
    return out
```

## Generator Constraints

ARC-GEN fixes the grid size at `15 x 15` and creates exactly three boxes. Widths are sampled from `1..5`, heights from `1..7`, and colors are a shuffle of `[1, 2, 4]`. Horizontal starts are sampled until the rectangles do not overlap with a margin of one column. Because every generated box touches the bottom row, the box height is recoverable by counting color cells upward from the bottom in its column span.

## Reference Notes

The ARC-DSL solver extracts colored objects, removes them with `cover`, then shifts each object by `-height(object)` rows using `toivec(invert(height(obj)))` and paints the shifted objects. The Code Golf solution transposes columns and uses the count of the bottom-cell color in each column to move the colored run upward, matching the per-box height rule.
