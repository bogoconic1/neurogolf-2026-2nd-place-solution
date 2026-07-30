# task375 Semantics

## Sources

- Current champion builder: `solutions_py/task375.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task375.json`
- ARC-GEN task id: `ea786f4a`
- ARC-DSL task id: `ea786f4a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ea786f4a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ea786f4a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task375.py`

## Pattern

The input is an odd-sized square filled with one nonzero color, except the center cell is black. The output keeps the same square and color but paints both diagonals black, forming an X from corner to corner. The center is already black in the input and remains black.

For an `n x n` input square, all cells `(r, c)` with `r == c` or `r + c == n - 1` become black. Every other cell keeps the input color. The submitted NeuroGolf graph must still emit the full dense `[1,10,30,30]` tensor with zeros outside the active square.

## Readable Python Solver

```python
def solve(grid):
    out = [list(row) for row in grid]
    n = len(out)
    for r in range(n):
        out[r][r] = 0
        out[r][n - 1 - r] = 0
    return out
```

## Generator Constraints

- `size` is an odd square side. Random generation uses `2 * randint(2, 7) + 1`, so random sizes are 5, 7, 9, 11, 13, and 15.
- The validator also includes explicit size 3, and the test example includes size 11.
- `color` is one random nonzero ARC color.
- The input and output are square and use only black plus that one color.
- The input center cell is black before transformation.
- The output blacks the main diagonal and anti-diagonal across the whole active square.
- There is no tie-breaking, object selection, or color inference beyond preserving the filled square color off the diagonals.

## Reference Notes

ARC-DSL computes the square width, builds the main diagonal from the origin and the anti-diagonal from the top-right corner, combines those two line sets, and fills them with zero. Code Golf 2025 expresses the same in-place loop by setting `row[i]` and `row[~i]` to zero for each row.

The ARC-DSL solver only fills diagonal cells with zero on top of the input, so any ONNX rewrite should prefer reusing the free input as the default output and only constructing a diagonal suppression mask.
