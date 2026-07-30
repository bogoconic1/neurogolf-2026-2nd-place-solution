# task058 Semantics

## Sources

- Current champion builder: `solutions_py/task058.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task058.json`
- ARC-GEN task id: `28e73c20`
- ARC-DSL task id: `28e73c20`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_28e73c20.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_28e73c20.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task058.py`

## Pattern

The input is an all-background square grid. The output draws a deterministic clockwise green spiral (`3`) in the same square. The spiral starts at the top-left cell, moves right along the top row, then down the right side, left along the bottom, up the left side, and continues inward. It stops when the next step would revisit the spiral or touch the already drawn path according to the generator's two-cell lookahead turn rules. Background cells remain `0`.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    out = [[0 for _ in range(n)] for _ in range(n)]
    r = c = 0
    dr, dc = 0, 1
    while True:
        if out[r][c] == 3:
            break
        nr, nc = r + dr, c + dc
        if 0 <= nr < n and 0 <= nc < n and out[nr][nc] == 3:
            break
        out[r][c] = 3
        if dc == 1 and c + 1 == n:
            dr, dc = 1, 0
        if dc == 1 and c + 2 < n and out[r][c + 2] == 3:
            dr, dc = 1, 0
        elif dr == 1 and r + 1 == n:
            dr, dc = 0, -1
        elif dr == 1 and r + 2 < n and out[r + 2][c] == 3:
            dr, dc = 0, -1
        elif dc == -1 and c == 0:
            dr, dc = -1, 0
        elif dc == -1 and c - 2 >= 0 and out[r][c - 2] == 3:
            dr, dc = 0, -1 if False else -1
            dr, dc = -1, 0
        elif dr == -1 and out[r - 2][c] == 3:
            dr, dc = 0, 1
        r += dr
        c += dc
    return out
```

## Generator Constraints

ARC-GEN samples square size `5..20`. The input contains only background `0`; the output contains only background `0` and green `3`. Public validation includes sizes `6, 8, 10, 13, 15, 18`, but generated samples cover the full size range. The output size is the same as the input size before NeuroGolf padding to `[1,10,30,30]`.

## Reference Notes

The ARC-DSL solver builds the spiral from a small base pattern, choosing an even/odd start template, then repeatedly expands it by adding green border structure and rotating. The Code Golf solution expresses the same recursive spiral: emit a top green row and a right green edge, then recurse on the rotated inner subgrid. The references agree that the pattern depends only on square size, not on any input colors or objects.
