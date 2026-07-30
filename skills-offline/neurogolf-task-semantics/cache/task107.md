# task107 Semantics

## Sources

- Current champion builder: `solutions_py/task107.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task107.json`
- ARC-GEN task id: `469497ad`
- ARC-DSL task id: `469497ad`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_469497ad.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_469497ad.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task107.py`

## Pattern

The input is always a `5x5` grid. It contains a `2x2` non-red box placed at
one of three top-left positions: `(0,1)`, `(1,0)`, or `(1,1)`. The last row and
last column contain matching color markers at symmetric positions
`(4-i,4)` and `(4,4-i)` for `i=0..4`. Repeated marker colors encode the scale:

```text
factor = number of distinct nonzero input colors - 1
```

Equivalently, the generator's factor is one plus the number of distinct marker
colors. The output first upscales every input cell by this factor. It then draws
red (`2`) diagonal corner markers around the upscaled `2x2` box: for each
`k=0..factor-1`, one red pixel is placed just outside each corner of the scaled
box, moving outward along the four diagonals. Red markers overwrite the upscaled
content at those coordinates.

## Readable Python Solver

```python
def solve(grid):
    n = 5
    colors = {v for row in grid for v in row}
    factor = len(colors) - 1  # includes black in colors

    out = [[0 for _ in range(n * factor)] for _ in range(n * factor)]
    for r in range(n):
        for c in range(n):
            for dr in range(factor):
                for dc in range(factor):
                    out[r * factor + dr][c * factor + dc] = grid[r][c]

    # Locate the 2x2 box by finding the nonzero 2x2 block away from the
    # bottom/right marker border.
    box_color = None
    box_r = box_c = None
    for r in range(4):
        for c in range(4):
            vals = [grid[r + dr][c + dc] for dr in (0, 1) for dc in (0, 1)]
            if vals[0] and vals.count(vals[0]) == 4:
                box_color = vals[0]
                box_r, box_c = r, c
                break
        if box_color is not None:
            break

    for k in range(factor):
        lo_r = box_r * factor - k - 1
        hi_r = (box_r + 2) * factor + k
        lo_c = box_c * factor - k - 1
        hi_c = (box_c + 2) * factor + k
        for rr, cc in ((lo_r, lo_c), (lo_r, hi_c), (hi_r, lo_c), (hi_r, hi_c)):
            if 0 <= rr < n * factor and 0 <= cc < n * factor:
                out[rr][cc] = 2
    return out
```

## Generator Constraints

The base grid size is fixed at `5`. The box top-left is one of `(0,1)`,
`(1,0)`, `(1,1)`. The box color is any non-red color. The marker colors exclude
red and the box color. The marker list has length 5, and each new marker either
introduces a new color or repeats the previous marker color, so the number of
distinct marker colors can vary. The output size is `5*factor`, where factor is
`2..6`, embedded in the standard `[1,10,30,30]` NeuroGolf tensor.

## Reference Notes

ARC-DSL computes the scale as `decrement(numcolors(I))`, upscales the whole input, finds the smallest object as the scaled box, shoots four diagonal rays from its upper-left and lower-left corners, underfills red, and then paints the red object. The Code Golf solution expresses the same repeated upscale and red corner-filling process through recursive transpose/string replacement tricks.
