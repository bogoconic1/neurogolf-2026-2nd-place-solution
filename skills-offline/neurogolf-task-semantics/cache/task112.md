# task112 Semantics

## Sources

- Current champion builder: `solutions_py/task112.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task112.json`
- ARC-GEN task id: `4938f0c2`
- ARC-DSL task id: `4938f0c2`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4938f0c2.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4938f0c2.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task112.py`

## Pattern

The input contains a 2x2 green block and a red pixel pattern placed diagonally outside that block. The red pattern is defined relative to one corner of the green block. In the common incomplete case, only the upper-left copy of the red pattern is present. The output keeps the original grid size and colors, then adds the missing red copies in the other three diagonal quadrants by reflecting the red pattern across the vertical and horizontal center lines of the 2x2 green block.

Equivalently, let the green block top-left corner be `(brow, bcol)`. A red pixel at `(brow - 1 - r, bcol - 1 - c)` implies red pixels at:

- `(brow - 1 - r, bcol + 2 + c)`
- `(brow + 2 + r, bcol - 1 - c)`
- `(brow + 2 + r, bcol + 2 + c)`

Inputs may already contain all reflected copies; in that case the output is the same as the input.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])

    green = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 3]
    brow = min(r for r, _ in green)
    bcol = min(c for _, c in green)

    out = [row[:] for row in grid]
    red = {(r, c) for r in range(h) for c in range(w) if grid[r][c] == 2}

    # The generator always places the defining pattern in the upper-left
    # quadrant. If showall=True, those pixels are still present, so this branch
    # handles both incomplete and already-complete inputs.
    seeds = [
        (r, c)
        for r, c in red
        if r < brow and c < bcol
    ]

    for sr, sc in seeds:
        dr = brow - 1 - sr
        dc = bcol - 1 - sc
        for rr, cc in (
            (brow - 1 - dr, bcol - 1 - dc),
            (brow - 1 - dr, bcol + 2 + dc),
            (brow + 2 + dr, bcol - 1 - dc),
            (brow + 2 + dr, bcol + 2 + dc),
        ):
            out[rr][cc] = 2

    return out
```

## Generator Constraints

- Original ARC grids have width and height in `[10, 30]`.
- The green block is always a 2x2 block, color `3`, with enough margin for the reflected pattern.
- The red pattern uses color `2` and is generated from a `length x length` coordinate set where `length` is `3` or `4`; `(0, 0)` is forced present and the final pattern has more than four pixels.
- The output always has all four diagonal reflected copies around the green block.
- With probability one quarter, `showall=True` and the input already contains all four red copies. Otherwise the input contains only the upper-left copy.
- There are no competing green blocks, no ambiguous colors, and no intended tie-breaking beyond the unique 2x2 green block.

## Reference Notes

- ARC-DSL finds objects, locates color `2`, mirrors the red set vertically and horizontally, shifts those reflected sets by the red set height/width plus the 2x2 green gap, and branches to identity when the input already has more than four objects. This matches the generator show-all identity case.
- The Code Golf solution is a compact double-rotation/zip trick. It effectively repeatedly reflects the grid around the green block location, filling the missing red copies.
- The references agree: the output is full-size, preserves all existing content, and only adds red reflected copies.
