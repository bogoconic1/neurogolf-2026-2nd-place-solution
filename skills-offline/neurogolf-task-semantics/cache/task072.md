# task072 Semantics

## Sources

- Current champion builder: `solutions_py/task072.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task072.json`
- ARC-GEN task id: `3428a4f5`
- ARC-DSL task id: `3428a4f5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3428a4f5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3428a4f5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task072.py`

## Pattern

The input is a `13x5` grid. Row `6` is a yellow separator. Rows `0..5` and
rows `7..12` are two aligned `6x5` binary red patterns on black background.
The output is a `6x5` grid: each cell is green when exactly one of the
corresponding top/bottom cells is red, and black when the two cells agree.
Equivalently, output the XOR or symmetric difference of red support in the top
half and red support in the bottom half.

## Readable Python Solver

```python
def solve(grid):
    h = 6
    w = 5
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            top_red = grid[r][c] == 2
            bottom_red = grid[r + h + 1][c] == 2
            if top_red != bottom_red:
                out[r][c] = 3
    return out
```

## Generator Constraints

- Input width is fixed at `5`.
- `height=6` is half the input height; input height is `2 * height + 1 = 13`.
- The middle row index `6` is always yellow (`4`) across all five columns.
- Top and bottom halves are independently sampled random red pixel sets.
- Red pixels use color `2`; output XOR pixels use green `3`.
- Output size is fixed at `6x5`.
- There is no object tie-breaking, color inference, translation, or variable
  geometry beyond the fixed top/bottom alignment.

## Reference Notes

ARC-DSL splits the grid into `tophalf(I)` and `bottomhalf(I)`, extracts red cells from both, computes their union minus their intersection, and fills those coordinates with green on a blank `6x5` canvas.

The Code Golf 2025 solution recursively compares corresponding top and bottom rows and multiplies inequality by `3`, matching the red-support XOR rule.
