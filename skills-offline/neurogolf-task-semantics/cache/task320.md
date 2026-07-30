# task320 Semantics

## Sources

- Current champion builder: `solutions_py/task320.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task320.json`
- ARC-GEN task id: `ce9e57f2`
- ARC-DSL task id: `ce9e57f2`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ce9e57f2.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ce9e57f2.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task320.py`

## Pattern

The input has four vertical red (`2`) bars in odd columns `1,3,5,7`. Each bar
starts at the bottom of the grid and has length from 2 to 10. The output keeps
the top half of each bar red and recolors the bottom `floor(length/2)` cells of
each bar to cyan (`8`). Background remains 0.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for c in range(1, w, 2):
        rows = [r for r in range(h) if grid[r][c] == 2]
        k = len(rows) // 2
        for r in rows[-k:]:
            out[r][c] = 8
    return out
```

## Generator Constraints

ARC-GEN samples four lengths independently in `[2, 10]`. The generated grid has
width 9 and height `max(lengths)+1`, then NeuroGolf pads it to 30x30. Bar cells
are exactly color 2; there are no other foreground colors. Bars are bottom
aligned, separated by blank columns, and all relevant rows fit within the first
11 rows after padding.

## Reference Notes

The ARC-DSL solver treats each red bar as an object, connects its upper-left corner to its center of mass, fills that segment with color 8, then switches 8 to 2. The generator is clearer numerically: recolor the lower half of each bar cyan. The Code Golf solution computes a per-column count and replaces the bottom half accordingly.
