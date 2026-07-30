# task095 Semantics

## Sources

- Current champion builder: `solutions_py/task095.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task095.json`
- ARC-GEN task id: `4258a5f9`
- ARC-DSL task id: `4258a5f9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4258a5f9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4258a5f9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task095.py`

## Pattern

The input is a fixed 9x9 black grid containing zero or more isolated gray `5`
pixels. The gray pixels are centers of non-overlapping 3x3 neighborhoods. The
output preserves each gray center and fills all eight neighboring cells around
each center with blue `1`, producing a blue 3x3 block with a gray center for
every marker. Empty cells remain black.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 5:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and (dr or dc):
                        out[rr][cc] = 1
    return out
```

## Generator Constraints

ARC-GEN uses a square grid with `size=9`. It samples up to 9 candidate marker
centers with rows and columns in `1..7`, rejecting any candidate whose 3x3 box
would overlap a previously accepted 3x3 box. This guarantees every marker has a
complete 3x3 neighborhood inside the grid and no two output 3x3 blocks overlap.
The random branch can accept fewer than 9 markers, including none in principle,
although bundled examples contain several markers. Colors are fixed: background
black `0`, marker center gray `5`, and filled neighbors blue `1`.

## Reference Notes

The ARC-DSL solver computes `ofcolor(I, FIVE)`, applies 8-neighborhood expansion with `neighbors`, and fills those neighbor cells with `ONE`. Since `neighbors` excludes the center, the original gray cells remain gray. The Code Golf solution is highly compressed, but agrees with the same local dilation around gray centers.
