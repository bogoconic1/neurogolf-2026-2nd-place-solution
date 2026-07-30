# task187 Semantics

## Sources

- Current champion builder: `solutions_py/task187.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task187.json`
- ARC-GEN task id: `7b6016b9`
- ARC-DSL task id: `7b6016b9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7b6016b9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7b6016b9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task187.py`

## Pattern

The input is a `20..25` by `20..25` grid containing one drawing color on a black background. The drawing color is never red (`2`) or green (`3`). It forms several hollow axis-aligned rectangles and line extensions on rectangle rows or columns. The output preserves every non-black drawing-color cell. Black cells connected to the outside border through 4-neighbor black paths become green (`3`). Black cells enclosed inside the colored rectangles, and therefore not connected to the border, become red (`2`). Equivalently, this is a binary flood-fill/background-vs-hole classification of black cells, followed by recoloring exterior black to green and enclosed black to red.

## Readable Python Solver

```python
from collections import deque

def solve(grid):
    h, w = len(grid), len(grid[0])
    exterior = [[False] * w for _ in range(h)]
    q = deque()

    def add(r, c):
        if 0 <= r < h and 0 <= c < w and grid[r][c] == 0 and not exterior[r][c]:
            exterior[r][c] = True
            q.append((r, c))

    for r in range(h):
        add(r, 0)
        add(r, w - 1)
    for c in range(w):
        add(0, c)
        add(h - 1, c)

    while q:
        r, c = q.popleft()
        add(r - 1, c)
        add(r + 1, c)
        add(r, c - 1)
        add(r, c + 1)

    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0:
                out[r][c] = 3 if exterior[r][c] else 2
    return out
```

## Generator Constraints

- Grid width and height are each sampled in `20..25`, then optionally flipped vertically and/or transposed.
- A single drawing color is sampled from all colors except red (`2`) and green (`3`).
- The generator always creates a main hollow rectangle and a smaller attached rectangle; it may also add a lower-right smaller rectangle and/or a right-side rectangle.
- Rectangle borders and random horizontal/vertical line extensions use the drawing color. Rectangle interiors are black in the input and red in the output.
- The output is initialized to green. The draw routine then writes the drawing color on borders/lines and red inside rectangle interiors, so exterior/background black becomes green.
- Random line extensions are rejected if they intersect each other in illegal ways. They may touch or extend from box borders but do not change the core rule: black components that border the canvas are exterior, enclosed black components are holes.

## Reference Notes

The ARC-DSL solver computes all univalued objects, filters to objects that do not border the grid, fills those non-bordering objects with red, then replaces remaining black with green. This is exactly the flood-fill interpretation above: non-bordering black components are holes; bordering black components are exterior background. The Code Golf 2025 solution is heavily golfed but repeatedly applies a local propagation/recoloring update, consistent with spreading exterior/green state through the black background and leaving enclosed cells to become red.
