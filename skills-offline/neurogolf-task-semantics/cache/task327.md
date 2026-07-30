# task327 Semantics

## Sources

- Current champion builder: `solutions_py/task327.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task327.json`
- ARC-GEN task id: `d13f3404`
- ARC-DSL task id: `d13f3404`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d13f3404.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d13f3404.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task327.py`

## Pattern

The input is a `3x3` grid containing three non-black colored pixels. The three colored pixels are guaranteed to lie on distinct down-right diagonals, i.e. no two active pixels have the same `col - row` value. The output is a `6x6` black grid. For each colored input pixel `(r, c)` with color `v`, draw a down-right ray of color `v` in the output starting at `(r, c)`:

```text
(r, c), (r+1, c+1), (r+2, c+2), ...
```

The ray continues while both coordinates are inside the `6x6` output. Because the generator enforces distinct diagonals, rays never conflict and no tie-breaking is needed. Black input cells do not contribute anything.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(2 * w)] for _ in range(2 * h)]
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color == 0:
                continue
            rr, cc = r, c
            while rr < 2 * h and cc < 2 * w:
                out[rr][cc] = color
                rr += 1
                cc += 1
    return out
```

## Generator Constraints

ARC-GEN defaults to `size=3`, so generated NeuroGolf examples have `3x3` inputs and `6x6` outputs. The random branch scans all cells in shuffled order, keeps the first cells whose `col - row` diagonal has not yet been used, and then zips those locations with `common.random_colors(3)`. In practice this creates exactly three colored source pixels with nonzero colors on distinct down-right diagonals. The explicit validation examples also use exactly three colored pixels. The output loop uses `range(2 * size - max(r, c))`, which is equivalent to drawing from `(r, c)` to the bottom or right edge of the `6x6` output.

Guaranteed invariants:

- input shape is `3x3`
- output shape is `6x6`
- exactly three nonzero colors are used
- source pixels occupy distinct down-right diagonals
- output rays are same-color copies of the corresponding source pixel
- background is black

The main unsafe shortcut would be assuming a fixed source count or diagonal order without respecting the diagonal uniqueness. Distinct diagonals make a single scalar color per output diagonal valid; if two colors could share a diagonal, the compact scalar diagonal formulation would be ambiguous.

## Reference Notes

The ARC-DSL solver treats each colored pixel as an object, applies `shoot` with direction `UNITY` from the object's center, recolors that ray with the object's color, and paints all rays onto a `6x6` black canvas. This directly matches the ARC-GEN output loop.

The Code Golf 2025 solution keeps a rolling previous-diagonal state while iterating rows extended by black padding. Its compact expression is another view of the same transformation: each output row is obtained from source values and shifted previous values so colors propagate down-right one step per output row.

There is no disagreement among the references. The important generator detail is the one-source-per-down-right-diagonal guarantee, which allows scalar diagonal color propagation instead of a full conflict-resolution graph.
