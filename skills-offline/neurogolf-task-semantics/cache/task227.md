# task227 Semantics

## Sources

- Current champion builder: `solutions_py/task227.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task227.json`
- ARC-GEN task id: `94f9d214`
- ARC-DSL task id: `94f9d214`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_94f9d214.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_94f9d214.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task227.py`

## Pattern

The input is a 4x8 grid made from two aligned 4x4 panels stacked vertically in the standard 30x30 one-hot tensor. The top panel contains green cells on a black background; the bottom panel contains blue cells on a black background. The output is a 4x4 grid where a cell is red exactly when the aligned top and bottom panel cells are both black. All other output cells are black.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(4)] for _ in range(4)]
    for r in range(4):
        for c in range(4):
            if grid[r][c] == 0 and grid[r + 4][c] == 0:
                out[r][c] = 2
    return out
```

## Generator Constraints

ARC-GEN always uses `size=4`, so inputs are 4 rows by 8 columns in ARC form, represented as two 4x4 halves. The top half uses only black and green (`3`); the bottom half uses only black and blue (`1`). Each half is populated by an independently sampled random set of pixels, so either color may appear in any subset of its panel. The output is always 4x4 and uses only black (`0`) and red (`2`). There is no tie-breaking or object geometry beyond aligned cellwise intersection of black positions.

## Reference Notes

The ARC-DSL solution takes the top half and bottom half, finds black cells in each, intersects those coordinate sets, and fills those positions red on a 4x4 black canvas. The Code Golf 2025 solution is a compact recursive/array expression equivalent to testing where the top and bottom halves are both zero and multiplying the boolean mask by red.
