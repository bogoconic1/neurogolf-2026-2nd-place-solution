# task221 Semantics

## Sources

- Current champion builder: `solutions_py/task221.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task221.json`
- ARC-GEN task id: `91413438`
- ARC-DSL task id: `91413438`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_91413438.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_91413438.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task221.py`

## Pattern

The input ARC grid is a 3x3 grid containing background color 0 and exactly one
nonzero foreground color. Let `z` be the number of zero cells and `k = 9 - z`
be the number of foreground cells. The output ARC grid is a square of size
`3*z` by `3*z`, viewed as a `z` by `z` grid of 3x3 tiles. Copy the original
3x3 foreground pattern into the first `k` tile slots in row-major order and
leave every other cell black. In the NeuroGolf dense representation this square
appears in the top-left of the standard `[1, 10, 30, 30]` output tensor, with
everything outside it left at zero.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    assert h == 3 and w == 3

    z = sum(1 for row in grid for v in row if v == 0)
    fg_cells = [(r, c, grid[r][c]) for r in range(3) for c in range(3)
                if grid[r][c] != 0]
    k = len(fg_cells)
    out = [[0 for _ in range(3 * z)] for _ in range(3 * z)]
    for tile_index in range(k):
        tr, tc = divmod(tile_index, z)
        for r, c, color in fg_cells:
            out[3 * tr + r][3 * tc + c] = color
    return out
```

## Generator Constraints

ARC-GEN always uses a 3x3 input. If parameters are not fixed, it samples
between 2 and 6 distinct foreground pixels from the 9 cells and chooses one
nonzero color with `common.random_color()`. Therefore `k` is 2..6 and `z` is
7..3. The output side length is `3*z`, so the generated ARC outputs are
9x9, 12x12, 15x15, 18x18, or 21x21. The foreground color is shared by all
nonzero pixels; there are no multiple-color cases and no tie-breaking between
objects.

## Reference Notes

ARC-DSL computes `z = colorcount(I, ZERO)`, constructs a wide padded copy of the input, shifts the foreground object by offsets `0, 3, 6, ...` for the first `k = 9 - z` tile positions, then splits the wide canvas into `z` pieces and merges them vertically. This is equivalent to row-major placement into a `z`-column tile grid. The Code Golf solution encodes the same rule by repeating each input row `z` times and using a shrinking row counter to produce exactly the first `k` tile copies. The references agree with ARC-GEN.
