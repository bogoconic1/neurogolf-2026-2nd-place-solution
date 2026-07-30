# task126 Semantics

## Sources

- Current champion builder: `solutions_py/task126.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task126.json`
- ARC-GEN task id: `54d82841`
- ARC-DSL task id: `54d82841`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_54d82841.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_54d82841.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task126.py`

## Pattern

The input is a small rectangular grid with black background. It contains one or more colored `2 x 3` shooter shapes. Each shooter has cells at `(r,c)`, `(r,c+1)`, `(r,c+2)`, `(r+1,c)`, and `(r+1,c+2)`, leaving the lower middle cell empty. The output preserves the input and adds a yellow (`4`) marker on the last row of the grid in the center column of each shooter, i.e. at `(height-1, c+1)`. Shooter colors can repeat or vary and are never yellow in generated cases.

## Readable Python Solver

```python
def solve(grid):
    H, W = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for c in range(1, W - 1):
        found = False
        for r in range(H - 1):
            cells = [grid[r][c - 1], grid[r][c], grid[r][c + 1], grid[r + 1][c - 1], grid[r + 1][c + 1]]
            if cells[0] and all(v == cells[0] for v in cells) and grid[r + 1][c] == 0:
                found = True
                break
        if found:
            out[H - 1][c] = 4
    return out
```

## Generator Constraints

ARC-GEN samples width `5..20` and height `5..10`. Shooters are placed from left to right starting at column `1`; each shape consumes columns `c..c+2`, and the next start column advances by `3` or `4` depending on color repetition, so shooters do not overlap. The top row of each shooter is sampled so the two-row shape fits above the bottom row. Colors are random non-yellow ARC colors; adjacent shooters may share the same color when spacing is `4`.

The generated output is the input plus yellow markers in the last row at every shooter center column. Hand-authored validation examples cover a single shooter, repeated colors, adjacent varying colors, and multiple heights.

## Reference Notes

The ARC-DSL solver extracts objects, takes each object's center, keeps the column coordinate (`last(center)`), pairs it with `height(input)-1`, and fills those bottom-row cells with yellow. The Code Golf solution transposes columns with `zip(*g)` and writes yellow in the last row for columns whose column sum is nonzero and where the nonzero color appears in that column. Both references indicate that only the shooter center columns matter; the original colored shapes are preserved.
