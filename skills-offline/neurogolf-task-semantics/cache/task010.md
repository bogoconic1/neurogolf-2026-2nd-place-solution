# task010 Semantics

## Sources

- Current champion builder: `solutions_py/task010.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task010.json`
- ARC-GEN task id: `08ed6ac7`
- ARC-DSL task id: `08ed6ac7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_08ed6ac7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_08ed6ac7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task010.py`

## Pattern

The input is a fixed 9x9 grid containing exactly four vertical gray bars (color 5). Bars occupy odd columns `1, 3, 5, 7`, but their left-to-right order is a permutation of four distinct heights. The output keeps the same bar cells but recolors each complete bar by its height rank: tallest becomes color 1, second tallest color 2, third color 3, shortest color 4. Background remains black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    cols = [c for c in range(w) if any(grid[r][c] == 5 for r in range(h))]
    heights = []
    for c in cols:
        rows = [r for r in range(h) if grid[r][c] == 5]
        heights.append((len(rows), c))
    rank_by_col = {c: rank + 1 for rank, (_, c) in enumerate(sorted(heights, reverse=True))}
    out = [[0 for _ in row] for row in grid]
    for height, c in heights:
        color = rank_by_col[c]
        for r in range(h - height, h):
            out[r][c] = color
    return out
```

## Generator Constraints

- Grid size is fixed at 9x9.
- Exactly four bars are generated.
- Bar columns are odd positions determined by `order[bar] * 2 + 1`, so active columns are always `1, 3, 5, 7` in some rank order.
- Heights are four distinct values sampled from 1..9 and sorted descending before being assigned to columns.
- Input bars are gray (`5`); output colors are rank colors `1, 2, 3, 4`.
- Background is black.

## Reference Notes

ARC-DSL orders the gray objects by height and paints each object with its rank from the interval of object count downwards. ARC-GEN confirms there are no ties, no extra objects, no variable size, and no non-bar shapes. The Code Golf solution exploits the fixed 9x9 layout and column sums to derive ranks.
