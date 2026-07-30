# task052 Semantics

## Sources

- Current champion builder: `solutions_py/task052.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task052.json`
- ARC-GEN task id: `25d8a9c8`
- ARC-DSL task id: `25d8a9c8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_25d8a9c8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_25d8a9c8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task052.py`

## Pattern

The input is a `3x3` color grid. Each row is either solid (all three cells have the same non-gray color) or mixed with exactly two cells of one color and one cell of a different color. The output is also `3x3`: a solid gray (`5`) row for every solid input row, and a solid black (`0`) row for every mixed input row.

## Readable Python Solver

```python
def solve(grid):
    return [[5 if len(set(row)) == 1 else 0 for _ in row] for row in grid]
```

## Generator Constraints

ARC-GEN uses size `3`. It chooses three or four random non-gray colors. It creates one or two solid rows, then fills the remaining rows with two colors in a 2:1 pattern, shuffles each mixed row, and shuffles row order. Hand examples cover different color choices and row orders. The output depends only on per-row equality, not on the actual colors.

## Reference Notes

The ARC-DSL solver finds horizontal objects of size 3, fills those row indices gray, and fills all other cells black. The Code Golf solution uses `len(set(row)) % 2 * 5`, which is `5` for one unique color and `0` for two unique colors under the generator.
