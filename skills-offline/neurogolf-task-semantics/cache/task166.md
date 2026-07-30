# task166 Semantics

## Sources

- Current champion builder: `solutions_py/task166.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task166.json`
- ARC-GEN task id: `6d75e8bb`
- ARC-DSL task id: `6d75e8bb`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6d75e8bb.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6d75e8bb.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task166.py`

## Pattern

The input contains one cyan object on a black background. In the untransformed generator view the object is a stack of left-aligned cyan horizontal strips, with lengths varying by row, plus one extra cyan pixel in a row above the next strip. The whole example may be flipped horizontally and/or transposed, so the visible orientation can be mirrored or vertical.

The output keeps all cyan cells unchanged and fills every black cell inside the cyan object's bounding box with red (`2`). Cells outside the bounding box remain black. The output grid has the same height and width as the input grid before NeuroGolf padding.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] != 0]
    if not cells:
        return [row[:] for row in grid]
    r0 = min(r for r, _ in cells)
    r1 = max(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    c1 = max(c for _, c in cells)
    out = [row[:] for row in grid]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if out[r][c] == 0:
                out[r][c] = 2
    return out
```

## Generator Constraints

- Colors are black background (`0`), cyan object (`8`), and red fill (`2`) in the output.
- `num_lengths` is `6..9`; `max_length` is `4..6`.
- Untransformed width is `max_length + 3..6`; height is `num_lengths + 2..5`.
- The cyan strip lengths list has `num_lengths` entries, each in `1..max_length`.
- The untransformed cyan/red bounding rectangle has height `num_lengths` and width `max_length`.
- The rectangle top-left `(brow, bcol)` is strictly inset: `brow` is `1..height-num_lengths-1`, and `bcol` is `1..width-max_length-1`.
- One extra cyan pixel is placed at row `brow + prow`, col `bcol + pcol`, with `prow in 0..num_lengths-2` and `pcol < lengths[prow+1]`; it may already be covered by that row's strip, but remains inside the rectangle.
- Optional horizontal flip and optional transpose are applied to both input and output, so optimization must be orientation-agnostic unless it explicitly handles those two flags.

## Reference Notes

The ARC-DSL solver takes the first foreground object, extracts its subgrid, replaces zeros in that subgrid with red, shifts it back to the object's upper-left corner, and paints it onto the original input. That is exactly a bounding-box fill of black cells around the cyan object.

The Code Golf solution implements the same row-wise fill effect compactly. It relies on comparing each row with the nonzero support/bounds, but the semantic result is still: cyan stays cyan, black cells inside the object bounding box become red.
