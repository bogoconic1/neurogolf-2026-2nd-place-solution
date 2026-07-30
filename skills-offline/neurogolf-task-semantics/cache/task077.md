# task077 Semantics

## Sources

- Current champion builder: `solutions_py/task077.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task077.json`
- ARC-GEN task id: `36fdfd69`
- ARC-DSL task id: `36fdfd69`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_36fdfd69.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_36fdfd69.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task077.py`

## Pattern

The input is a noisy grid with a single static color plus several hidden
rectangles. Red `2` cells are visible cells inside the rectangles. Yellow `4`
cells from the true output are hidden in the input by replacing them with the
static color. The output restores every hidden rectangle: for each rectangle,
fill every non-red cell inside its bounding box with yellow `4`, preserving red
`2` cells and preserving all noise/static cells outside the rectangles.

The red cells in a rectangle may be disconnected because some rectangle cells
are hidden as static color, but the generator guarantees each rectangle row and
column has at least one visible red cell. Rectangles are separated, so grouping
red cells by local proximity recovers each hidden rectangle.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    reds = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 2]

    # Red cells from the same generated rectangle can be separated by hidden
    # yellow/static cells, so use the same kind of bounded proximity closure as
    # the ONNX label-propagation champion: cells within a 5x5 neighborhood are
    # connected, transitively.
    remaining = set(reds)
    groups = []
    while remaining:
        start = remaining.pop()
        group = {start}
        stack = [start]
        while stack:
            r, c = stack.pop()
            linked = [p for p in remaining if abs(p[0] - r) <= 2 and abs(p[1] - c) <= 2]
            for p in linked:
                remaining.remove(p)
                group.add(p)
                stack.append(p)
        groups.append(group)

    out = [row[:] for row in grid]
    for group in groups:
        rows = [r for r, _ in group]
        cols = [c for _, c in group]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r][c] != 2:
                    out[r][c] = 4
    return out
```

## Generator Constraints

Inputs have height 15 to 20 and width equal to height plus `-1`, `0`, or `1`.
There is one static/noise color, chosen from colors other than red `2` and
yellow `4`. Random static pixels are placed first. Then 2 to 4 non-overlapping
rectangles are generated with width 2 to 7 and height 2 to 3. Rectangles are
placed with a separation margin of 2.

Inside each rectangle, cells that were empty become red `2`; cells that already
held the static color become yellow `4` in the output, but are shown as the
static color in the input. The generator rejects rectangles unless every row and
every column has at least one visible red cell, so the red-cell bounding box is
the true rectangle box.

## Reference Notes

ARC-DSL upscales the grid by 2, extracts red objects, connects sufficiently separated red fragments through rectangle deltas, fills those deltas with yellow, repaints red cells, and downscales. This expresses the same bounding-box completion rule while avoiding direct component heuristics on the original grid.

The Code Golf solution repeatedly rewrites the grid by detecting red/yellow-like neighborhood relations in rotated views. It reinforces that the true operation is rectangle completion around red evidence, with the static color acting as the hidden yellow value in the input.
