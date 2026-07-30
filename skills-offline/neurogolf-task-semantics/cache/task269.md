# task269 Semantics

## Sources

- Current champion builder: `solutions_py/task269.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task269.json`
- ARC-GEN task id: `ac0a08a4`
- ARC-DSL task id: `ac0a08a4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ac0a08a4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ac0a08a4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task269.py`

## Pattern

The input is always a `3x3` grid. Between `1` and `9` cells are non-black and each non-black cell has an arbitrary ARC color. Let `k` be the number of non-black cells. The output is the input grid upscaled by factor `k`: every input cell becomes a `k x k` solid block of the same color, so the output size is `3k x 3k`. Black input cells become black blocks.

## Readable Python Solver

```python
def solve(grid):
    k = sum(1 for row in grid for v in row if v != 0)
    out = []
    for row in grid:
        expanded_row = []
        for v in row:
            expanded_row.extend([v] * k)
        for _ in range(k):
            out.append(expanded_row[:])
    return out
```

## Generator Constraints

The generator uses a fixed `3x3` input. It samples a non-empty subset of the nine cells, with count `k` from `1..9`, assigns random nonzero colors to those selected cells, and leaves all other cells black. The output is produced by `grid_enhance(size=3, len(colors)=k, ...)`, so output dimensions range from `3x3` when `k=1` to `27x27` when `k=9`. There are no tie cases: the upscale factor is exactly the nonzero-cell count.

## Reference Notes

ARC-DSL computes `colorcount(I, ZERO)`, subtracts that from `9`, and calls `upscale(I, k)`. The Code Golf solution expands the serialized grid by repeating columns and rows according to the number of nonzero entries. Both references agree that colors are copied unchanged and that only the count of non-black cells controls scale.
