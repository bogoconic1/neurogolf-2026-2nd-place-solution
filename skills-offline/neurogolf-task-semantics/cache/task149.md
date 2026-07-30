# task149 Semantics

## Sources

- Current champion builder: `solutions_py/task149.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task149.json`
- ARC-GEN task id: `6773b310`
- ARC-DSL task id: `6773b310`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6773b310.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6773b310.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task149.py`

## Pattern

The input is always an `11x11` black grid partitioned by cyan (`8`) separator lines into a `3x3` array of `3x3` mini-grids. Each mini-grid contains either one or two pink (`6`) pixels. The output is a `3x3` grid: output cell `(r, c)` is blue (`1`) if the corresponding input mini-grid `(r, c)` contains exactly two pink pixels, otherwise it is black (`0`). Cyan separator lines are structural only and do not appear in the output.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(3)] for _ in range(3)]
    for br in range(3):
        for bc in range(3):
            count = 0
            r0 = br * 4
            c0 = bc * 4
            for dr in range(3):
                for dc in range(3):
                    if grid[r0 + dr][c0 + dc] == 6:
                        count += 1
            if count == 2:
                out[br][bc] = 1
    return out
```

## Generator Constraints

`minisize` is fixed at 3. The input shape is fixed `11x11`: three `3x3` content blocks in each direction with one-cell cyan separator rows and columns between blocks. For every one of the nine mini-grids, ARC-GEN samples either one or two distinct pink pixels from the `3x3` block. At least one mini-grid has two pink pixels. Inputs use only colors `0`, `6`, and `8`; outputs use only `0` and `1`. Output shape is always `3x3`.

## Reference Notes

ARC-DSL compresses away the cyan separator rows and columns, examines each centered `3x3` neighborhood corresponding to a mini-grid, filters those objects whose pink count is two, fills those positions blue, replaces pink with black, and downscales by three. The Code Golf solution recursively scans every fourth row group and uses byte offsets for the three mini-grid columns to test whether the pink count crosses the two-pixel threshold. All references agree that the only signal is the number of pink cells in each mini-grid, not their arrangement.
