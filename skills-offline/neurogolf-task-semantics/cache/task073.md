# task073 Semantics

## Sources

- Current champion builder: `solutions_py/task073.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task073.json`
- ARC-GEN task id: `3618c87e`
- ARC-DSL task id: `3618c87e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3618c87e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3618c87e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task073.py`

## Pattern

The input is a fixed `5x5` grid with a gray base row at the bottom. One or two
"towers" are present. In each tower column, row `3` is gray and row `2` is a
single blue cell. The output moves each blue singleton down two rows onto the
bottom base row in the same column. The original blue cells disappear, the gray
support cells remain gray, and all other bottom-row cells remain gray.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == 1:
                out[r][c] = 0
                out[r + 2][c] = 1
    return out
```

## Generator Constraints

- Grid size is fixed at `5x5`.
- The bottom row is always gray (`5`) in input and output.
- Tower columns are sampled from one or two columns. If there are two towers,
  their columns are non-adjacent (`abs(c0 - c1) > 1`).
- In each tower column, input row `3` is gray and input row `2` is blue.
- In the output, the blue cell moves to row `4` in the same column, replacing
  that base cell. Row `3` remains gray; row `2` becomes black.
- There is no variable color choice, object size, rotation, or translation
  other than the fixed `(+2, 0)` move of blue singletons.

## Reference Notes

ARC-DSL finds connected objects, filters singleton objects, merges them, and moves those singleton cells by `TWO_BY_ZERO`. Under the generator, the only singleton foreground objects are the blue caps, because gray support/base cells are connected to each other.

The Code Golf 2025 solution encodes the same fixed geometry compactly: keep the first three rows black, keep the gray support row, and compute the bottom row by moving/toggling the blue cap columns.

It uses signed logits and the scorer's `output > 0` threshold rather than explicit one-hot construction.
