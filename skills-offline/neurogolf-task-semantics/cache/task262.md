# task262 Semantics

## Sources

- Current champion builder: `solutions_py/task262.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task262.json`
- ARC-GEN task id: `a85d4709`
- ARC-DSL task id: `a85d4709`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a85d4709.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a85d4709.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task262.py`

## Pattern

The input is a 3x3 black grid with exactly one gray cell (`5`) in each row. The gray cell's column chooses the output row color from a fixed palette by column: column 0 maps to color 2, column 1 maps to color 4, and column 2 maps to color 3. The output is a 3x3 grid where each row is filled entirely with the color selected by that row's gray-cell column.

## Readable Python Solver

```python
def solve(grid):
    palette = [2, 4, 3]
    out = []
    for row in grid:
        col = row.index(5)
        out.append([palette[col]] * 3)
    return out
```

## Generator Constraints

- Input and output are always 3x3.
- Each input row contains exactly one gray cell and otherwise black/background cells.
- The column list is sampled by shuffling `range(3)` in random generation, so generated rows usually use the three columns as a permutation; validation examples also include repeated columns such as `[2,2,2]` and `[0,1,0]`.
- The output color palette is fixed as `(2,4,3)` indexed by the gray column.
- There are no other colors, objects, rotations, or size branches.

## Reference Notes

The ARC-DSL solver collects the gray cell coordinates and filters by last coordinate/column. It fills full horizontal frontiers for column 0 with color 2, column 2 with color 3, and column 1 with color 4. The Code Golf solution encodes the same row-wise lookup in a compact arithmetic expression over the three input row values.
