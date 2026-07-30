# task254 Semantics

## Sources

- Current champion builder: `solutions_py/task254.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task254.json`
- ARC-GEN task id: `a61f2674`
- ARC-DSL task id: `a61f2674`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a61f2674.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a61f2674.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task254.py`

## Pattern

The input is a 9x9 bar chart. Gray (`5`) vertical bars rise upward from the bottom row. Bars occupy alternating columns with horizontal stride 2. The first bar column is either column 0 or column 1 (`offset` 0 or 1). There are 4 bars when offset is 1 and either 4 or 5 bars when offset is 0. Bar heights are distinct values sampled from 1..9.

The output is a blank 9x9 grid except for two recolored bars at their original locations. The tallest gray bar is recolored blue (`1`). The shortest gray bar is recolored red (`2`). All medium-height bars are removed to black (`0`). There are no ties because heights are sampled without replacement.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    bars = []
    for c in range(w):
        rows = [r for r in range(h) if grid[r][c] == 5]
        if rows:
            bars.append((len(rows), c, rows))

    out = [[0 for _ in range(w)] for _ in range(h)]
    if not bars:
        return out

    min_h = min(height for height, _, _ in bars)
    max_h = max(height for height, _, _ in bars)
    for height, c, rows in bars:
        if height == max_h:
            color = 1  # blue
        elif height == min_h:
            color = 2  # red
        else:
            continue
        for r in rows:
            out[r][c] = color
    return out
```

## Generator Constraints

The grid is always 9x9. The input contains only black background and gray bars. Bars are one cell wide, bottom-aligned, and placed at columns `offset + 2*i`. `offset` is 0 or 1. If `offset == 1`, exactly 4 bars fit in columns 1,3,5,7. If `offset == 0`, either 4 bars in columns 0,2,4,6 or 5 bars in columns 0,2,4,6,8. Heights are unique samples from 1 through 9, so exactly one shortest and one tallest bar exist. Output colors are only black, blue, and red.

## Reference Notes

The ARC-DSL solver extracts connected objects, selects `argmax(size)` and `argmin(size)`, replaces all gray with black, then paints the largest object blue (`1`) and smallest object red (`2`). This confirms that object size equals bar height and that no tie handling is needed. The Code Golf solution computes column sums from the bottom-aligned bars and uses arithmetic to emit only max/min bars.
