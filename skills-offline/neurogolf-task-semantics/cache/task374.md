# task374 Semantics

## Sources

- Current champion builder: `solutions_py/task374.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task374.json`
- ARC-GEN task id: `ea32f347`
- ARC-DSL task id: `ea32f347`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ea32f347.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ea32f347.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task374.py`

## Pattern

The input is a 10x10 grid containing exactly three non-overlapping gray line objects. Each object is a straight horizontal or vertical segment. Their lengths are distinct and sorted by role: the shortest line becomes color `2`, the middle-length line becomes color `4`, and the longest line becomes color `1`. Background remains black.

The transformation is purely object recoloring by size. Geometry, orientation, and position are preserved.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    objects = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 5 or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 5:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            objects.append(cells)

    ordered = sorted(objects, key=len)
    out = [row[:] for row in grid]
    for row in out:
        for c, value in enumerate(row):
            if value == 5:
                row[c] = 4
    for color, cells in ((2, ordered[0]), (1, ordered[-1])):
        for r, c in cells:
            out[r][c] = color
    return out
```

## Generator Constraints

- Grid size is fixed at 10x10.
- There are exactly three gray line objects in the input.
- Lengths are sampled as three distinct values from 2..9 and sorted increasing.
- Each line is either vertical (`up=1`) or horizontal (`up=0`).
- Placement is rejected until the three lines do not overlap and have spacing around same-orientation or crossing cases.
- Output colors are assigned by sorted length: shortest `2`, middle `4`, longest `1`.
- The input uses only black background and gray `5`; output uses black plus colors `1`, `2`, and `4`.

## Reference Notes

ARC-DSL extracts objects, replaces all gray with color 4 as the default middle-line color, then fills the smallest object with 2 and the largest object with 1. Code Golf 2025 uses a compact recursive transpose/replacement trick, but it is implementing the same length-rank recoloring.

There is no need to identify absolute length values; only the relative smallest and largest objects matter. Because line objects are straight and non-overlapping, row/column run evidence is enough if it preserves orientation and separates objects.
