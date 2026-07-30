# task176 Semantics

## Sources

- Current champion builder: `solutions_py/task176.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task176.json`
- ARC-GEN task id: `7447852a`
- ARC-DSL task id: `7447852a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7447852a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7447852a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task176.py`

## Pattern

The input is a 3-row grid of width 5..25. A red (`2`) path runs left to right in a vertical bounce pattern over rows `0,1,2,1,0,1,2,1,...`. The output keeps the red path and fills selected black cells with yellow (`4`). Yellow fill alternates by phase of the bounce endpoints: for some top endpoint phases, cells below the red cell in that column are yellow; for some bottom endpoint phases, cells above the red cell in that column are yellow; other columns remain black except for red.

Equivalently, view the black connected components between/around the red path in reading/order-by-center order. Fill every third black object, starting from the first, with yellow.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    mode = -1
    for c in range(w):
        r = c % (2 * h - 2)
        r = r if r < h else 2 * h - r - 2
        if r in (0, h - 1):
            mode = (mode + 1) % 6
        if mode in (0, 5):
            for rr in range(r + 1, h):
                if out[rr][c] == 0:
                    out[rr][c] = 4
        if mode in (2, 3):
            for rr in range(0, r):
                if out[rr][c] == 0:
                    out[rr][c] = 4
    return out
```

## Generator Constraints

The height is fixed at 3 in all generated data. Width is random 5..25, with fixed validation widths 10, 15, 18, and 25. Colors are only black/background `0`, red `2`, and yellow `4`. The red path is deterministic from column index; there is no noise and no alternate orientation.

## Reference Notes

The ARC-DSL solver identifies black objects, orders them by center, selects object indices contained in `range(0, count, 3)`, and fills those selected black components with yellow. The Code Golf solution tracks a phase counter while scanning the red path and emits the same every-third fill pattern.
