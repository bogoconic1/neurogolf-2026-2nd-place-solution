# task286 Semantics

## Sources

- Current champion builder: `solutions_py/task286.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task286.json`
- ARC-GEN task id: `b782dc8a`
- ARC-DSL task id: `b782dc8a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b782dc8a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b782dc8a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task286.py`

## Pattern

The input is a rectangular maze-like grid with black corridors (`0`), cyan walls/background (`8`), and two non-cyan seed colors. One seed color appears at a single corridor cell and the other seed color appears on one or more orthogonally adjacent corridor cells. The output preserves the walls, unrelated black cells, and seed cells, then flood-fills the connected black corridor component reachable from the seeds with the two seed colors alternating by checkerboard parity.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    parity_color = {}
    queue = []
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v not in (0, 8):
                parity_color[(r + c) & 1] = v
                queue.append((r, c))

    seen = set(queue)
    head = 0
    while head < len(queue):
        r, c = queue[head]
        head += 1
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or grid[nr][nc] != 0:
                continue
            seen.add((nr, nc))
            out[nr][nc] = parity_color[(nr + nc) & 1]
            queue.append((nr, nc))
    return out
```

## Generator Constraints

The random generator uses width and height from 10 through 25. It starts with cyan (`8`), carves a black maze/path network on one checkerboard parity with two-cell corridor steps, then places two distinct non-cyan seed colors on adjacent empty corridor cells. The generated output flood-fills from those colored cells through black cells, using the seed color assigned to each `(row + col) % 2` parity. Bundled ARC-AGI examples are included alongside ARC-GEN examples and must also be handled; the random generator alone does not cover every bundled case.

## Reference Notes

ARC-DSL uses `leastcolor` to identify the singleton seed color, finds the neighboring other seed color, selects the black object adjacent to that other color, and splits that black component by even/odd Manhattan distance from the singleton seed. The Code Golf solution performs a recursive/parity flood fill.
