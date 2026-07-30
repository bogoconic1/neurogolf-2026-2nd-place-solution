# task183 Semantics

## Sources

- Current champion builder: `solutions_py/task183.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task183.json`
- ARC-GEN task id: `77fdfe62`
- ARC-DSL task id: `77fdfe62`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_77fdfe62.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_77fdfe62.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task183.py`

## Pattern

The input is a square grid of size 6x6, 8x8, or 10x10. A blue (`1`) frame surrounds an even-sized interior: the blue frame lies on rows/columns `1` and `n-2`, while the four outer corners carry four arbitrary non-blue, non-cyan colors. Inside the framed square, cyan (`8`) marks selected cells and black (`0`) marks unselected cells.

The output is the framed interior with the blue border and outer corner shell removed. Its size is therefore `2x2`, `4x4`, or `6x6`. Every cyan cell in the interior is recolored according to which quadrant it belongs to: top-left quadrant uses the top-left corner color, top-right uses the top-right corner color, bottom-left uses the bottom-left corner color, and bottom-right uses the bottom-right corner color. Interior black cells remain black.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    n = h - 4
    half = n // 2
    corners = {
        (0, 0): grid[0][0],
        (0, 1): grid[0][h - 1],
        (1, 0): grid[h - 1][0],
        (1, 1): grid[h - 1][h - 1],
    }
    out = [[0 for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            if grid[r + 2][c + 2] == 8:
                out[r][c] = corners[(1 if r >= half else 0, 1 if c >= half else 0)]
    return out
```

## Generator Constraints

ARC-GEN chooses `size = 2 * randint(1, 3)`, so the output size is always `2`, `4`, or `6`, and the input size is always `size + 4` (`6`, `8`, or `10`). The frame geometry is fixed: blue lines at row/column `1` and `size + 2`, with four colored corner cells at `(0,0)`, `(0,size+3)`, `(size+3,0)`, and `(size+3,size+3)`.

The four corner colors are sampled from colors excluding blue (`1`) and cyan (`8`). The interior cyan cells are a non-empty random subset of the `size x size` interior. There is no tie-breaking or object ordering: every selected cyan cell maps independently to its quadrant corner color, and every black interior cell maps to black. Padding outside the output remains the standard zero-hot NeuroGolf padding.

## Reference Notes

The ARC-DSL solver extracts the subgrid bounded by color `8`, removes blue and cyan from the input, compresses the remaining four-corner layout, upscales that compressed corner-color pattern by half the output width, and then clears locations corresponding to black cells in the original cyan subgrid. This matches the generator rule: quadrant colors are broadcast over an output canvas and masked by the cyan interior pattern.

The Code Golf 2025 solution encodes the same recursive structure tersely: it uses the outer rows/corners to determine the repeated quadrant colors and removes the two-cell blue/corner shell. The three references agree that the only dynamic state is output size, four corner colors, and the cyan mask.
