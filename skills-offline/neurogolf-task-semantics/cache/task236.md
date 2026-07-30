# task236 Semantics

## Sources

- Current champion builder: `solutions_py/task236.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task236.json`
- ARC-GEN task id: `99b1bc43`
- ARC-DSL task id: `99b1bc43`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_99b1bc43.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_99b1bc43.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task236.py`

## Pattern

The input is a fixed 9x4 grid made from two 4x4 panels separated by a full yellow row. The upper panel contains blue cells, the lower panel contains red cells. The output is a 4x4 grid with green (`3`) wherever exactly one of the corresponding upper or lower panel cells is occupied, and black elsewhere. This is the XOR / symmetric difference of the two binary masks after aligning the lower panel back up by 5 rows.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for c in range(4):
            top = grid[r][c] != 0
            bottom = grid[r + 5][c] != 0
            if top ^ bottom:
                out[r][c] = 3
    return out
```

## Generator Constraints

- Grid size is fixed: width `4`, height `9`.
- Row `4` is entirely yellow (`4`) and is only a separator.
- Top-panel selected cells are blue (`1`) in rows `0..3`.
- Bottom-panel selected cells are red (`2`) in rows `5..8`, corresponding to output rows `0..3` after subtracting 5.
- `common.random_pixels(4, 4)` independently samples top and bottom pixel sets; either set may be sparse or dense.
- The output is always exactly 4x4 and uses only colors `0` and `3`.

## Reference Notes

- ARC-DSL computes zero-cell sets in the top and bottom halves, takes their symmetric difference, and fills those positions green. Symmetric difference of zero masks is equivalent to XOR of occupied masks.
- The Code Golf one-liner is a compact recursive/vectorized XOR between the upper rows and lower rows, scaled by green.
- There is no dynamic size, color selection, object ordering, or tie-breaking.
