# task078 Semantics

## Sources

- Current champion builder: `solutions_py/task078.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task078.json`
- ARC-GEN task id: `3906de3d`
- ARC-DSL task id: `3906de3d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3906de3d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3906de3d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task078.py`

## Pattern

The task is a 10x10 column-bar sorting problem using only black `0`, blue `1`,
and red `2`. In each active column, blue cells already occupy the top segment.
Red cells appear at the bottom of the input column. The output moves those red
cells upward so that each active column is sorted vertically as black cells
above? No: after sorting by color order, the visible result is blue `1` cells at
the top, red `2` cells immediately below the blue segment, and black `0` cells
below them. Empty columns remain black.

Equivalently, for each column, let `A` be the count of blue cells and `B` be the
count of blue-or-red cells in rows 0..9. Output color is:

- blue `1` for rows `0 <= r < A`
- red `2` for rows `A <= r < B`
- black `0` for rows `B <= r < 10`

Rows and columns outside the 10x10 task grid are zero-hot padding under the
NeuroGolf tensor contract.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        blues = sum(1 for r in range(h) if grid[r][c] == 1)
        reds = sum(1 for r in range(h) if grid[r][c] == 2)
        for r in range(blues):
            out[r][c] = 1
        for r in range(blues, blues + reds):
            out[r][c] = 2
    return out
```

## Generator Constraints

ARC-GEN always uses a square size `10`. The active column block starts at
`col` in `[1, 3]` and ends before `size - loc`, where `loc` is also in `[1, 3]`.
For every active column, `top` is 1..5. If `top >= 4`, `bottom` is 0; otherwise
`bottom` is 1..6. Input and output both have the same blue top segment. Input
places red cells at the bottom of each active column, while output places the
same number of red cells directly below the blue top segment.

The task uses only colors 0, 1, and 2. The maximum useful spatial extent is the
true 10x10 grid; padded rows and columns in the `[1,10,30,30]` NeuroGolf tensor
are zero-hot, not black channel 0.

## Reference Notes

ARC-DSL rotates the grid, switches colors 1 and 2, orders each row, switches the colors back, then mirrors. This is a compact way to sort each original column by its colored cells. The Code Golf solution transposes/rotates and sorts each column with a key that puts blue before red before black, producing the same count-based column reconstruction.
