# task231 Semantics

## Sources

- Current champion builder: `solutions_py/task231.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task231.json`
- ARC-GEN task id: `963e52fc`
- ARC-DSL task id: `963e52fc`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_963e52fc.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_963e52fc.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task231.py`

## Pattern

The input is a height-5 grid with a small horizontal periodic color pattern drawn in one or two adjacent rows. The pattern row block starts at vertical offset 1 or 2, has height 1 or 2, and has period width 2 or 3. The pattern is repeated across the true input width. The output has the same height and double the input width, extending the same row pattern to the right. Rows outside the pattern block remain black.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(2 * w)] for _ in range(h)]
    for r, row in enumerate(grid):
        # The generator period is 2 or 3, so the first six cells contain a whole
        # number of periods. Repeating that prefix is enough for widths up to 10.
        prefix = row[:6]
        repeated = (prefix * 4)[:2 * w]
        out[r] = repeated
    return out
```

## Generator Constraints

- Input height is fixed at `5`.
- Input width is sampled from `6..10`; output width is exactly `2 * width`.
- Pattern width `wide` is `2` or `3`; pattern height `tall` is `1` or `2`.
- Vertical offset is `1` or `2`, so the drawn rows stay within the height-5 grid.
- Two random colors are chosen and every generated pattern contains both colors.
- The pattern is drawn for columns `0..2*width-1`; drawing outside the input grid is clipped, while the output keeps the full doubled width.

## Reference Notes

- ARC-DSL computes the horizontal period of the input object, crops one period-height block, rotates/repeats it enough times, rotates back, and crops to height by double-width.
- The Code Golf 2025 solution repeats `row[:6]` and crops to `2 * len(row)`, exploiting that periods are only 2 or 3 and widths are at most 10.
- Black rows have a zero prefix, so the same row-prefix repeat rule naturally leaves them black.
