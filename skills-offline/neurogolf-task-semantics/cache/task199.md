# task199 Semantics

## Sources

- Current champion builder: `solutions_py/task199.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task199.json`
- ARC-GEN task id: `834ec97d`
- ARC-DSL task id: `834ec97d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_834ec97d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_834ec97d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task199.py`

## Pattern

The input is a square black grid containing exactly one non-yellow colored pixel. If that pixel is at row `r`, column `c`, with color `k`, the output removes it from its original location, moves the same color down one row to `(r + 1, c)`, and fills rows `0..r` with yellow (`4`) in every column whose parity matches `c`. All other cells remain black.

The selected pixel is never yellow in the generator. Its row is at most `size - 2`, so the downward move is always in bounds. The yellow parity fill includes the original pixel's row and column, so the original colored pixel becomes yellow in the output.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    r0 = c0 = color = None
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v != 0:
                r0, c0, color = r, c, v
                break
        if color is not None:
            break

    for r in range(r0 + 1):
        for c in range(c0 % 2, w, 2):
            out[r][c] = 4
    out[r0 + 1][c0] = color
    return out
```

## Generator Constraints

- Grid is square with `size = randint(3, 15)`.
- There is exactly one input colored pixel.
- Pixel row is `0..size-2`; pixel column is `0..size-1`.
- Pixel color is a random ARC color excluding yellow (`4`); black is not used for the pixel.
- Output size equals input size. NeuroGolf output remains the dense `[1,10,30,30]` padded canvas.
- The downward move is always in bounds.
- Yellow parity columns start at `col % 2` and step by `2`, up to but not including `size`.

## Reference Notes

The ARC-DSL solver finds the single object, shifts it down, blanks the original pixel, then fills yellow above the shifted pixel at columns in an interval from `col-10` to `col+10` with step `2`; clipping to the grid makes that equivalent to all columns matching the original column parity.

The Code Golf 2025 solution encodes the same rule compactly: locate the nonzero row, create a parity yellow row prefix using the pixel column parity, then append the suffix containing the moved colored pixel.
