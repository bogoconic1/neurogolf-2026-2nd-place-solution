# task288 Semantics

## Sources

- Current champion builder: `solutions_py/task288.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task288.json`
- ARC-GEN task id: `b8cdaf2b`
- ARC-DSL task id: `b8cdaf2b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b8cdaf2b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b8cdaf2b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task288.py`

## Pattern

The input is a square black-background icon with two non-black colors. One color is the shirt/body color and the least frequent non-black color is the antenna color. The bottom row has shirt-color shoulder blocks on the left and right and an antenna-color neck segment in the center. The row above has a shirt-color neck segment. The output keeps the input and fills the missing antenna-color diagonal arms rising up-left and up-right from the cells just above the two ends of the bottom antenna segment.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0) + 1
    antenna = min(counts, key=counts.get)
    out = [row[:] for row in grid]

    bottom = grid[-1]
    cols = [c for c, v in enumerate(bottom) if v == antenna]
    left, right = min(cols), max(cols)

    r, c = h - 2, left
    while r >= 0 and c >= 0:
        if out[r][c] == 0:
            out[r][c] = antenna
        r -= 1
        c -= 1

    r, c = h - 2, right
    while r >= 0 and c < w:
        if out[r][c] == 0:
            out[r][c] = antenna
        r -= 1
        c += 1
    return out
```

## Generator Constraints

The generator chooses neck width `1` or `3`, shoulder width `1..3`, and two distinct non-black colors. The square side is `neck + 2*shoulder`, so sizes are odd values from 3 through 9 depending on the pair. The bottom row contains shirt shoulders and an antenna-color center neck; the row above contains a shirt-color neck. The output fills only black cells on the two antenna diagonals. Known task examples confirm the input omits those diagonal antenna cells even though the local ARC-GEN source appears to assign them to both `grid` and `output`.

## Reference Notes

ARC-DSL uses `leastcolor(I)` as the antenna color, shifts its cells up by one row, takes the upper-left and upper-right corners, shoots rays up-left and up-right, and underfills those rays with the antenna color. The Code Golf solution derives the shoulder width from zero cells on the row above the bottom and writes the two symmetric antenna diagonals.
