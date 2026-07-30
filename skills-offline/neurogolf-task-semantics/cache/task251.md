# task251 Semantics

## Sources

- Current champion builder: `solutions_py/task251.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task251.json`
- ARC-GEN task id: `a5313dff`
- ARC-DSL task id: `a5313dff`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a5313dff.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a5313dff.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task251.py`

## Pattern

The input is a square grid, generated at size 8..12, containing one to four red rectangular frame-like objects. Boxes may be fully inside the grid or clipped by the top/left/bottom/right border. The red pixels form outer borders and sometimes inner red cores; black pixels form holes/gaps inside these red structures.

The output preserves all original red and black pixels except for black connected components that are completely enclosed and do not touch the grid border. Those enclosed black components are recolored blue (`1`). Black components connected to the outside border remain black. In short: flood-fill the exterior black background from the border, then paint every remaining black cell blue.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    exterior = [[False] * w for _ in range(h)]
    stack = []

    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] == 0 and not exterior[r][c]:
                exterior[r][c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] == 0 and not exterior[r][c]:
                exterior[r][c] = True
                stack.append((r, c))

    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and not exterior[nr][nc]:
                exterior[nr][nc] = True
                stack.append((nr, nc))

    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not exterior[r][c]:
                out[r][c] = 1
    return out
```

## Generator Constraints

ARC-GEN chooses 1..4 boxes, grid size 8..12, widths and heights 4..6, and row/column origins from `-1` through the largest value that can keep or slightly clip a box. Boxes are rejected if they overlap under `common.overlaps(..., margin=-1)`. Red is the only foreground color. For each box, the generator draws a red rectangle, clears the one-cell interior ring to black, and draws a smaller red core inset by two cells. Fully contained boxes have their one-cell black ring turned blue in the output. Clipped boxes have holes connected to the canvas border and are not filled blue. The generator rejects samples where no blue would be drawn and rejects accidental extra red rectangles.

## Reference Notes

The ARC-DSL solver collects black objects, filters out black objects that border the input grid, merges the remaining black cells, and fills them with blue. This confirms the semantic rule is enclosed-black-hole filling, not rectangle parameter recovery. The Code Golf solution implements the same repeated exterior/background propagation idea in compressed form.
