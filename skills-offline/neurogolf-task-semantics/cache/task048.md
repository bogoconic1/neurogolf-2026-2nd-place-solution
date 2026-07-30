# task048 Semantics

## Sources

- Current champion builder: `solutions_py/task048.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task048.json`
- ARC-GEN task id: `239be575`
- ARC-DSL task id: `239be575`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_239be575.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_239be575.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task048.py`

## Pattern

The input is a small grid, width and height from 5 to 8, containing two red
2x2 boxes and scattered cyan pixels. Treat red and cyan cells as foreground and
use 4-neighbor connectivity. If the two red boxes are connected through a
foreground path, the output is a 1x1 cyan grid (`8`). If they are not connected,
the output is a 1x1 black grid (`0`).

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    red = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 2]
    if not red:
        return [[0]]

    start = red[0]
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] != 0:
                seen.add((nr, nc))
                stack.append((nr, nc))

    connected_red = sum(1 for cell in red if cell in seen)
    return [[8 if connected_red == len(red) else 0]]
```

## Generator Constraints

ARC-GEN samples grid width and height independently from 5..8. It places random
cyan pixels at about 40% density, then places exactly two non-overlapping red
2x2 boxes. The boxes do not overlap even with a one-cell margin. The generator
rejects ambiguous cases where 4-neighbor and 8-neighbor connectivity disagree,
so diagonal-only contact is not the deciding edge case. Hand examples cover
5x5, 7x5, 7x6, 6x6, and 6x8/8-height shapes.

## Reference Notes

The ARC-DSL solver finds foreground objects and filters objects whose palette contains red (`2`). If more than one red-containing foreground object exists, the boxes are disconnected and the output is black. Otherwise the red boxes are part of one foreground object and the output is cyan. The Code Golf solution is a hash-style classifier over the small static benchmark distribution, but the ARC-GEN reference makes the intended rule clear: foreground connectivity between the two red boxes.
