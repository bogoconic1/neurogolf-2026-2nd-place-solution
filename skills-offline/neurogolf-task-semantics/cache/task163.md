# task163 Semantics

## Sources

- Current champion builder: `solutions_py/task163.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task163.json`
- ARC-GEN task id: `6d0160f0`
- ARC-DSL task id: `6d0160f0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6d0160f0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6d0160f0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task163.py`

## Pattern

The grid is an 11x11 Hollywood-squares board: a 3x3 array of 3x3 mini-grids separated by gray lines. Each mini-grid contains several colored pixels. Exactly one pixel is yellow color 4. The yellow pixel identifies both the source mini-grid and the destination mini-grid. Copy the entire source mini-grid containing the yellow pixel into the destination mini-grid whose macro row/column are the yellow pixel's local row/column inside the source mini-grid. Clear all other mini-grid contents, preserving the gray separator lattice.

## Readable Python Solver

```python
def solve(grid):
    out = [[5 if r in (3, 7) or c in (3, 7) else 0 for c in range(11)] for r in range(11)]
    yr = yc = None
    for r in range(11):
        for c in range(11):
            if grid[r][c] == 4:
                yr, yc = r, c
    src_mr, src_mc = yr // 4, yc // 4
    dst_mr, dst_mc = yr % 4, yc % 4
    for rr in range(3):
        for cc in range(3):
            val = grid[src_mr * 4 + rr][src_mc * 4 + cc]
            if val != 5:
                out[dst_mr * 4 + rr][dst_mc * 4 + cc] = val
    return out
```

## Generator Constraints

`minisize=3`, so the visible board is always 11x11 with separator rows/columns at 3 and 7. For every macro tile `(r,c)`, the generator places 2..4 colored pixels at random local positions. Colors exclude gray 5 and yellow 4, then one random pixel among all mini-grid contents is changed to yellow. The output contains only the gray separator lattice and the copied source 3x3 tile at the destination macro tile selected by the yellow local coordinate.

## Reference Notes

The ARC-DSL solver locates color 4, maps the yellow absolute row/column to source macro coordinates by thresholds 3 and 7, crops the source 3x3 tile, replaces gray with black inside the crop, and shifts that object by four times the yellow local coordinate. The Code Golf solution uses the same relation with `p%3*4 + y//4` for source lookup and `p&-4 | x//4` / local terms to route the copied tile.
