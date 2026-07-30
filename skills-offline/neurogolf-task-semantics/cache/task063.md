# task063 Semantics

## Sources

- Current champion builder: `solutions_py/task063.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task063.json`
- ARC-GEN task id: `2bee17df`
- ARC-DSL task id: `2bee17df`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_2bee17df.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_2bee17df.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task063.py`

## Pattern

The input is an even square grid of size 10, 12, or 14. Red (`2`) and cyan (`8`) runs enter from the four borders with lengths sampled by the generator. The output keeps all existing colored cells and fills green (`3`) through any fully empty interior row or fully empty interior column. Filling is an underfill: only black cells are changed to green, while red/cyan cells remain unchanged.

Equivalently, for the active square size `n`, inspect rows and columns excluding the outer border. If every cell in `row[1:n-1]` is black, fill that row with green in black cells. If every cell in `col[1:n-1]` is black, fill that column with green in black cells.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    out = [row[:] for row in grid]
    free_rows = [all(grid[r][c] == 0 for c in range(1, n - 1)) for r in range(n)]
    free_cols = [all(grid[r][c] == 0 for r in range(1, n - 1)) for c in range(n)]
    for r in range(n):
        if free_rows[r]:
            for c in range(n):
                if out[r][c] == 0:
                    out[r][c] = 3
    for c in range(n):
        if free_cols[c]:
            for r in range(n):
                if out[r][c] == 0:
                    out[r][c] = 3
    return out
```

## Generator Constraints

The generator samples `size = 2 * randint(5, 7)`, so sizes are 10, 12, and 14. It creates colored runs from all four borders, walking clockwise around the perimeter; each run length is `max(1, randint(0, 3))` in random generation, though public examples may include zero in the fixed validation seeds. The first `numred` perimeter runs are red and the rest are cyan. Optional vertical flip and transpose may be applied. Output fills green rows/columns only after the red/cyan border-run grid is built.

## Reference Notes

The ARC-DSL solver computes row and column predicates by splitting the grid and its 90-degree rotation: a row/column is fillable when the count of black cells equals `size - 2`, i.e. the interior excluding border cells is entirely black. It then creates horizontal and vertical frontiers and underfills them with green. The Code Golf solution performs the same row/column emptiness check compactly using zipped rows/columns.
