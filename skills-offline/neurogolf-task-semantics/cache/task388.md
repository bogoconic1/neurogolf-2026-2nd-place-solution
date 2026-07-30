# task388 Semantics

## Sources

- Current champion builder: `solutions_py/task388.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task388.json`
- ARC-GEN task id: `f5b8619d`
- ARC-DSL task id: `f5b8619d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f5b8619d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f5b8619d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task388.py`

## Pattern

The input is a square `2x2` to `6x6` black grid with one non-cyan foreground color placed in `1..size` cells. The output is a `2*size x 2*size` grid. It repeats the transformed input in a `2x2` tile: every column that contains a foreground pixel is filled with cyan (`8`) down the full height, and the original foreground pixels overwrite those cyan guide columns. The same columns repeat in the right tile, and the whole pattern repeats in the bottom tile.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    color = max((v for row in grid for v in row if v not in (0, 8)), default=0)
    cols = {c for r, row in enumerate(grid) for c, v in enumerate(row) if v == color}
    base = [[0 for _ in range(n)] for _ in range(n)]
    for c in cols:
        for r in range(n):
            base[r][c] = 8
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == color:
                base[r][c] = color
    return [row + row for row in base] + [row + row for row in base]
```

## Generator Constraints

ARC-GEN chooses square size `2..6`. It samples between `1` and `size` colored cells from the square, so at least one guide column exists and at most `size` colored cells exist. The foreground color is random excluding cyan. The generated output is `2*size` in both dimensions, then NeuroGolf embeds it in the standard dense `[1,10,30,30]` output tensor.

## Reference Notes

ARC-DSL identifies the least/non-background input color, gets those cells, builds vertical frontiers through them, underfills cyan along those columns without overwriting the original colored cells, then horizontally and vertically concatenates the result with itself. The Code Golf 2025 solution expresses the same rule row-wise: for each cell, keep the color if present, otherwise draw cyan when the column has any color, then duplicate rows and columns.
