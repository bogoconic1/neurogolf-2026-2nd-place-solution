# task215 Semantics

## Sources

- Current champion builder: `solutions_py/task215.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task215.json`
- ARC-GEN task id: `8eb1be9a`
- ARC-DSL task id: `8eb1be9a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8eb1be9a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8eb1be9a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task215.py`

## Pattern

The input contains one horizontal row of repeated copies of a small Conway sprite/tetris-like shape. The sprite is 3 rows tall, has width 2 or 3 depending on the generated shape, and uses one nonzero color. The copies tile left-to-right across the input width. When `flip=1`, every other horizontal tile is vertically flipped inside its 3-row band.

The output repeats that whole horizontal sprite band vertically with period 3. In ARC-GEN this is done by shifting the detected object band by multiples of its height (`3`) for offsets `-6,-3,0,3,6,9,...`, clipped to the grid. Existing input pixels are preserved; missing period bands above and below are painted in the same color and same horizontal phase/flip pattern.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    pts = [(r, c, v) for r, row in enumerate(grid) for c, v in enumerate(row) if v]
    if not pts:
        return out
    r0 = min(r for r, _, _ in pts)
    color = pts[0][2]
    band = [(r - r0, c) for r, c, _ in pts]
    for shift in range(-6, h + 6, 3):
        for dr, c in band:
            r = r0 + dr + shift
            if 0 <= r < h and 0 <= c < w:
                out[r][c] = color
    return out
```

## Generator Constraints

- Input width and height are random from 10 to 20.
- The sprite comes from `common.conway_sprite(common.randint(2, 3), 3)`, so it is exactly 3 rows tall and has width 2 or 3.
- The horizontal copies repeat every sprite width from column 0.
- The visible band offset is 3 or 4.
- One nonzero color is used.
- If `flip=1`, odd horizontal tiles are vertically flipped within the 3-row band.
- Output repeats the same band every 3 rows from `offset-6` through the grid height.

## Reference Notes

ARC-DSL identifies the foreground object, computes its height, shifts it by multiples of that height over the interval `-2..3`, and paints all shifted copies. The Code Golf solution uses a row-wise max with rows `3:6` and `6:9` repeated, reflecting the period-3 vertical tiling.
