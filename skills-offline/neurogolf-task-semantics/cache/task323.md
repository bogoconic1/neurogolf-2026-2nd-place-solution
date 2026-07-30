# task323 Semantics

## Sources

- Current champion builder: `solutions_py/task323.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task323.json`
- ARC-GEN task id: `d06dbe63`
- ARC-DSL task id: `d06dbe63`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d06dbe63.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d06dbe63.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task323.py`

## Pattern

The active grid is a 13x13 square embedded at the top-left of the standard
30x30 NeuroGolf canvas. It contains exactly one cyan (`8`) cell on an otherwise
zero background. The output preserves that cyan marker and draws gray (`5`)
cells in two opposing stair-step rays from the marker: one ray moves two rows
up, then two columns right, repeating; the other moves two rows down, then two
columns left, repeating. Cells outside the 13x13 active square remain zero.

## Readable Python Solver

```python
def solve(grid):
    h = 13
    out = [row[:] for row in grid]
    marker = None
    for r in range(h):
        for c in range(h):
            if grid[r][c] == 8:
                marker = (r, c)
                break
        if marker is not None:
            break

    row, col = marker
    out[row][col] = 8
    for dr, dc in [(-1, 1), (1, -1)]:
        r, c = row, col
        vertical = 2
        horizontal = 0
        while True:
            if vertical:
                r += dr
                vertical -= 1
                if r < 0 or r >= h:
                    break
                out[r][c] = 5
                if vertical == 0:
                    horizontal = 2
            else:
                c += dc
                horizontal -= 1
                if c < 0 or c >= h:
                    break
                out[r][c] = 5
                if horizontal == 0:
                    vertical = 2
    return out
```

## Generator Constraints

ARC-GEN uses `size=13` by default and places one cyan cell at a uniformly
chosen row and column in `[0, 12]`. The input and output are square 13x13
grids before NeuroGolf padding. The only nonzero input color is `8`; the only
new output color is gray `5`. Rays stop immediately when the next vertical or
horizontal step would leave the 13x13 square, so boundary markers may create
short or empty rays on one side. There are no multiple markers, no random
colors, no variable size in the bundled generator calls, and no tie-breaking
between objects.

## Reference Notes

The ARC-DSL solver constructs a small shape around the cyan center, repeats it at offsets `(-2, 2) * k` for `k=0..4`, then rotates the partially filled image to handle the opposite ray. The Code Golf 2025 solution uses repeated regex replacement on the string form and its reverse; that confirms the pattern is a deterministic text/stride geometry rather than object classification. The generator is clearer about the exact state machine: two vertical cells, then two horizontal cells, repeated in each direction.
