# task028 Semantics

## Sources

- Current champion builder: `solutions_py/task028.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task028.json`
- ARC-GEN task id: `1bfc4729`
- ARC-DSL task id: `1bfc4729`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1bfc4729.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1bfc4729.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task028.py`

## Pattern

The input is a 10x10 black grid with exactly two non-black pixels. One pixel in row 2 supplies the top color, and one pixel in row 7 supplies the bottom color; their columns do not affect the output. The output is a fixed 10x10 frame pattern on black: rows 0 and 2 are completely filled with the top color, rows 1, 3, and 4 contain the top color only in columns 0 and 9, rows 7 and 9 are completely filled with the bottom color, and rows 5, 6, and 8 contain the bottom color only in columns 0 and 9.

## Readable Python Solver

```python
def solve(grid):
    n = 10
    top = max(grid[2])
    bottom = max(grid[7])
    out = [[0 for _ in range(n)] for _ in range(n)]

    for r in range(5):
        out[r][0] = top
        out[r][n - 1] = top
    for c in range(n):
        out[0][c] = top
        out[2][c] = top

    for r in range(5, 10):
        out[r][0] = bottom
        out[r][n - 1] = bottom
    for c in range(n):
        out[7][c] = bottom
        out[9][c] = bottom

    return out
```

## Generator Constraints

ARC-GEN fixes `size=10`. It chooses two distinct nonzero colors and two independent columns in the closed range 2..7. The input then contains only `grid[2][cols[0]] = colors[0]` and `grid[7][cols[1]] = colors[1]`; every other cell is black. The output dimensions match the input dimensions. Because the colored pixels are away from the border and the two colors are distinct, there is no overlap or tie-breaking ambiguity.

## Reference Notes

ARC-DSL expresses the rule as independent top-half and bottom-half fills: take the least nonzero color in the top half and bottom half, fill the full 10-wide bounding row plus the horizontal frontier at row 2 for the top half, mirror that shape vertically, and replace the top color with the bottom color. Code Golf uses fixed row bytes `b"/ /  pp\x7fp\x7f"`: bytes whose value modulo 15 selects input row 2 or 7, and whose low bits either pass the color into all eight inner columns or zero them, while the two edge columns are always colored. These references agree that the input columns are irrelevant.
