# task332 Semantics

## Sources

- Current champion builder: `solutions_py/task332.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task332.json`
- ARC-GEN task id: `d406998b`
- ARC-DSL task id: `d406998b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d406998b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d406998b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task332.py`

## Pattern

The input is a 3-row grid with width 10 to 20. Each column contains exactly one gray pixel (`5`) and all other cells are black. The output keeps the same shape and recolors alternating gray pixels to green (`3`). The parity is counted from the right edge: a gray pixel becomes green when its column parity differs from the grid width parity. Equivalently, for even widths the odd-indexed columns become green, and for odd widths the even-indexed columns become green. All other gray pixels remain gray and black cells remain black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 5 and (c % 2) != (w % 2):
                out[r][c] = 3
    return out
```

## Generator Constraints

ARC-GEN fixes height to 3 and samples width uniformly from 10 to 20. For each column it chooses one row in `[0, 2]` and places a single gray pixel there. Thus every active column has exactly one gray cell, there are no other colors, and valid output width equals the number of gray columns. NeuroGolf tensors still use the standard padded `[1,10,30,30]` contract, so columns beyond the generated width must remain no-support rather than decoded black.

## Reference Notes

The ARC-DSL solution vertically mirrors the grid, selects gray cells whose mirrored column coordinate has even parity, fills them green, then mirrors back. This is the same as recoloring alternating columns from the right edge. The Code Golf solution destructively pops row values while using remaining row length parity to choose whether a gray cell maps to green or stays gray.
