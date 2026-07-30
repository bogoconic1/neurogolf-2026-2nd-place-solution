# task037 Semantics

## Sources

- Current champion builder: `solutions_py/task037.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task037.json`
- ARC-GEN task id: `1f876c06`
- ARC-DSL task id: `1f876c06`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1f876c06.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1f876c06.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task037.py`

## Pattern

The effective ARC grid is `10x10`. The input contains several nonzero foreground colors, each appearing exactly as the two endpoints of one diagonal line segment. A segment can slope down-right or down-left. The output is the same size and fills every cell on each inclusive diagonal segment with that segment color, leaving all other cells background `0`.

The NeuroGolf tensor contract wraps the 10x10 ARC grid in the standard `[1,10,30,30]` one-hot output; rows and columns outside the top-left 10x10 ARC extent are background.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    by_color = {}
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v:
                by_color.setdefault(v, []).append((r, c))
    for color, pts in by_color.items():
        if len(pts) != 2:
            continue
        (r0, c0), (r1, c1) = sorted(pts)
        dr = r1 - r0
        if dr == 0 or abs(c1 - c0) != dr:
            continue
        dc = 1 if c1 > c0 else -1
        for k in range(dr + 1):
            out[r0 + k][c0 + dc * k] = color
    return out
```

## Generator Constraints

ARC-GEN uses `size=10`. It samples `3..6` candidate diagonals. Each candidate has a maximum length `3..7`, random starting row, random starting column, and slope `+1` or `-1`. The accepted length grows until the diagonal would collide with an already occupied diagonal cell or form a corner-touch conflict with the previous row; candidates shorter than `3` are skipped. Accepted diagonals get distinct random colors.

The input keeps only the first and last cell of each accepted diagonal. The output contains the full segment. Segment lengths are therefore `3..7`, and same-color endpoint pairs are guaranteed to lie on a 45-degree diagonal.

## Reference Notes

The ARC-DSL solver partitions foreground objects, connects each object first and last cell, recolors the generated line by the object color, and paints the lines over the input. This confirms that the intended operation is per-color endpoint connection, not a global scalar diagonal fill.

The Code Golf 2025 solution repeatedly uses a regex over the reversed grid to replace zeros that lie between matching endpoint colors on a 35-character diagonal stride, then recurses until no changes remain. This reinforces the same endpoint-to-endpoint diagonal fill and shows that repeated propagation is a compact alternative expression for the rule.
