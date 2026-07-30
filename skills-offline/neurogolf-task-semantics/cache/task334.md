# task334 Semantics

## Sources

- Current champion builder: `solutions_py/task334.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task334.json`
- ARC-GEN task id: `d4469b4b`
- ARC-DSL task id: `d4469b4b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d4469b4b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d4469b4b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task334.py`

## Pattern

The input is a 5x5 grid containing black background and pixels of exactly one foreground color from `{1, 2, 3}`. The output is always a 3x3 grid using black `0` and gray `5`. The foreground color selects a fixed gray glyph:

- color `1`: a plus sign, with middle row and middle column gray.
- color `2`: a top T shape, with top row and middle column gray.
- color `3`: a lower-right L shape, with bottom row and right column gray.

The locations and count of foreground pixels in the input are irrelevant except that at least one nonzero foreground pixel is present and all foreground pixels share the same color.

## Readable Python Solver

```python
def solve(grid):
    fg = max(max(row) for row in grid)
    if fg == 1:
        rows, cols = {1}, {1}
    elif fg == 2:
        rows, cols = {0}, {1}
    elif fg == 3:
        rows, cols = {2}, {2}
    else:
        rows, cols = set(), set()
    return [[5 if r in rows or c in cols else 0 for c in range(3)] for r in range(3)]
```

## Generator Constraints

ARC-GEN samples a 5x5 square input. It chooses 9 to 16 distinct foreground cell positions and one foreground color from `[1, 2, 3]`, then writes that color into every selected cell. The output grid is a 3x3 square. There are no mixed foreground colors, no empty foreground case in generated tasks, and no dependence on input geometry beyond the selected foreground color. Training and test examples follow the same size and single-color constraints.

## Reference Notes

The ARC-DSL solution obtains the nonzero palette color, branches on whether it is `1` or `2`, and selects one of three row/column frontier centers: `UNITY` for color 1, `TWO_BY_TWO` for color 2, and `RIGHT` for color 3. It then combines the vertical and horizontal frontiers and fills those cells gray in a black 3x3 canvas. The Code Golf 2025 solution is an equivalent lookup keyed by `max(max(g))`, returning the corresponding 3x3 glyph.
