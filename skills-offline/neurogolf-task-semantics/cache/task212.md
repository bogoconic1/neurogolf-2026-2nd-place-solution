# task212 Semantics

## Sources

- Current champion builder: `solutions_py/task212.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task212.json`
- ARC-GEN task id: `8d510a79`
- ARC-DSL task id: `8d510a79`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8d510a79.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8d510a79.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task212.py`

## Pattern

The input is a 10x10 grid with a full horizontal gray row (`5`) called the horizon. All other nonzero cells are sparse source pixels colored blue (`1`) or red (`2`). Sources appear only at least one row away from the horizon: above sources are in rows `0..horizon-2`, and below sources are in rows `horizon+2..9`.

The output keeps the gray horizon row. Each source pixel creates a vertical ray in its own column:

- Blue (`1`) rays move away from the horizon. A blue source above the horizon fills upward through row `0`; a blue source below the horizon fills downward through row `9`.
- Red (`2`) rays move toward the horizon. A red source above the horizon fills downward until the cell immediately above the gray row; a red source below the horizon fills upward until the cell immediately below the gray row.

Rays write their source color starting at the source cell and stop at the grid boundary or at the already-filled gray horizon. Under the generator there is at most one source above and one source below in each column, so same-side ray collisions do not occur.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]

    horizon = next(r for r, row in enumerate(grid) if all(v == 5 for v in row))
    for c in range(w):
        out[horizon][c] = 5

    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color not in (1, 2):
                continue
            if color == 1:
                dr = -1 if r < horizon else 1
            else:
                dr = 1 if r < horizon else -1
            rr = r
            while 0 <= rr < h and out[rr][c] == 0:
                out[rr][c] = color
                rr += dr
    return out
```

## Generator Constraints

- Grid size is fixed at `10x10` before NeuroGolf padding to the standard 30x30 tensor.
- The gray horizon row is sampled from rows `3..6` and every cell in that row is gray `5`.
- For each column independently, the generator may create one source above the horizon and may create one source below the horizon.
- Above-source rows are sampled from `0..horizon-2`; below-source rows are sampled from `horizon+2..9`. The rows adjacent to the horizon are initially empty.
- Each source color is independently sampled as blue `1` or red `2`.
- Only colors `0`, `1`, `2`, and `5` appear.

## Reference Notes

The ARC-DSL solution finds color-1 and color-2 source cells and the gray row. It computes whether each source is above the horizon and then uses `shoot` with a vertical direction. Color 1 uses the direction away from the horizon; color 2 uses the direction toward the horizon, filtered to stay on the same side before filling. The Code Golf 2025 solution recursively scans rows and carries a 10-column state, toggling ray propagation behavior when the gray horizon is encountered.
