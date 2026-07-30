# task002 Semantics

## Sources

- Current champion builder: `solutions_py/task002.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task002.json`
- ARC-GEN task id: `00d62c1b`
- ARC-DSL task id: `00d62c1b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_00d62c1b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_00d62c1b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task002.py`

## Pattern

The input is a square grid, size 6..20 in generated data, using black `0` and green `3`. The output preserves all green cells and recolors to yellow `4` every black cell that is enclosed by green walls and does not have a 4-connected path to the grid border through black cells. Black background components that touch any border remain black. This includes the interiors of the generated green rectangular frames and any additional black regions closed off by overlapping/touching green structure.

## Readable Python Solver

```python
from collections import deque

def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]
    seen = [[False] * w for _ in range(h)]
    q = deque()

    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] == 0 and not seen[r][c]:
                seen[r][c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] == 0 and not seen[r][c]:
                seen[r][c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and not seen[nr][nc]:
                seen[nr][nc] = True
                q.append((nr, nc))

    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 and not seen[r][c]:
                out[r][c] = 4
    return out
```

## Generator Constraints

ARC-GEN samples between 1 and 8 non-overlapping rectangular green frames on a square grid of size 6..20. Frame widths and heights are 3..8, and each frame has green walls and a black interior in the input. The output turns every frame interior yellow. The generator also scatters random green cells with about 5% density before drawing frames; those static green cells can further divide black space. After drawing frames, it performs a final `is_surrounded` pass over black cells in the output, so the true rule is connected-component enclosure rather than only rectangle interiors.

Generated inputs contain only colors 0 and 3. Generated outputs contain 0, 3, and 4. Rectangles do not overlap under the generator's `overlaps(..., -1)` check, but they can be close; static green pixels can create additional enclosed pockets. Sizes are at most 20, embedded in the standard NeuroGolf 30x30 canvas.

## Reference Notes

The ARC-DSL solver finds black objects, filters out those that border the grid, and fills the remaining black cells with color 4. The Code Golf 2025 solution implements repeated directional propagation from the border/background view, which is equivalent to flood-filling exterior black and then recoloring the complement. There is no semantic disagreement: fill black components that do not touch the border.
