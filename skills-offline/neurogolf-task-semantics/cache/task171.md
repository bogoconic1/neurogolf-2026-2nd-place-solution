# task171 Semantics

## Sources

- Current champion builder: `solutions_py/task171.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task171.json`
- ARC-GEN task id: `6f8cd79b`
- ARC-DSL task id: `6f8cd79b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6f8cd79b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6f8cd79b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task171.py`

## Pattern

The input is an all-black rectangular grid. The output has the same height and width, with every border cell recolored cyan (`8`) and every interior cell left black (`0`). For a 3x3 grid this is the usual cyan frame around one black center; for larger rectangles it is a one-cell-thick cyan frame.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                out[r][c] = 8
    return out
```

## Generator Constraints

Generated grids have width and height independently sampled from 3 through 9. The input grid is all zero. The output keeps the same dimensions and sets only the top row, bottom row, left column, and right column to cyan. The bundled validation examples cover several rectangles, including 3x3, 3x4, 4x5, 6x5, and 6x7.

## Reference Notes

The ARC-DSL solver fills every cell that borders the input grid with color 8. The Code Golf 2025 solution recursively rotates/augments the grid until the border is filled; it is equivalent to drawing a cyan frame. The sources agree.
