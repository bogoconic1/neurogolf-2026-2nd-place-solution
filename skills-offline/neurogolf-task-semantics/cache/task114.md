# task114 Semantics

## Sources

- Current champion builder: `solutions_py/task114.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task114.json`
- ARC-GEN task id: `49d1d64f`
- ARC-DSL task id: `49d1d64f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_49d1d64f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_49d1d64f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task114.py`

## Pattern

The input is a tiny 2x2, 2x3, 3x2, or 3x3 colored grid using colors from `{1,2,3,4,8}` plus possible black. The output adds a one-cell border around the input. The input is copied into the center. The top border repeats the input's first row, the bottom border repeats the last row, the left border repeats the first column, and the right border repeats the last column. The four output corners remain black/background.

For a height `H` and width `W`, output shape is `(H+2, W+2)`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w + 2)] for _ in range(h + 2)]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            out[r + 1][c + 1] = color
            if r == 0:
                out[0][c + 1] = color
            if r == h - 1:
                out[h + 1][c + 1] = color
            if c == 0:
                out[r + 1][0] = color
            if c == w - 1:
                out[r + 1][w + 1] = color
    return out
```

## Generator Constraints

- Width and height are independently 2 or 3.
- Colors are drawn from `(1,2,3,4,8)`.
- A 3x3 input has its center forced to black.
- Output corners are always black because only edge-adjacent border cells are filled.
- There are no dynamic objects beyond the tiny grid extent; the only branch is whether each axis has size 2 or 3.

## Reference Notes

- ARC-DSL creates a canvas of shape `input_shape + 2`, shifts the input by `(1,1)`, then paints each non-corner border cell with the nearest shifted input cell by Manhattan distance.
- Code Golf explicitly constructs `[0] + top_row + [0]`, then left/right duplicated middle rows, then `[0] + bottom_row + [0]`.
- References agree that corners remain black rather than duplicating diagonal corner colors.
