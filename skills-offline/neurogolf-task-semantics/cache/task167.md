# task167 Semantics

## Sources

- Current champion builder: `solutions_py/task167.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task167.json`
- ARC-GEN task id: `6e02f1e3`
- ARC-DSL task id: `6e02f1e3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6e02f1e3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6e02f1e3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task167.py`

## Pattern

The input is always a 3x3 grid whose cell values are drawn from colors `2`, `3`, and `4`. The output is a black 3x3 grid containing exactly three gray (`5`) cells. The selected gray line depends only on the number of distinct colors in the input:

- 1 distinct color: gray horizontal line across the top row `(0,0),(0,1),(0,2)`.
- 2 distinct colors: gray main diagonal `(0,0),(1,1),(2,2)`.
- 3 distinct colors: gray anti-diagonal `(0,2),(1,1),(2,0)`.

Input positions and actual color identities do not matter beyond the distinct-color count.

## Readable Python Solver

```python
def solve(grid):
    n = len({v for row in grid for v in row})
    out = [[0 for _ in range(3)] for _ in range(3)]
    if n == 1:
        coords = [(0, 0), (0, 1), (0, 2)]
    elif n == 2:
        coords = [(0, 0), (1, 1), (2, 2)]
    else:
        coords = [(0, 2), (1, 1), (2, 0)]
    for r, c in coords:
        out[r][c] = 5
    return out
```

## Generator Constraints

ARC-GEN samples a nonempty subset of abstract color indices `{0,1,2}` with size 1..3, fills all nine cells by drawing from that subset, then adds `color_offset=2`, so only physical colors `2`, `3`, and `4` appear. Grid size is fixed at 3x3. Output uses only black and gray. The number of distinct sampled indices controls the output line exactly.

## Reference Notes

The ARC-DSL solver computes `numcolors(I)`, creates a black 3x3 canvas, then branches on whether the count equals 3 or 2 to choose endpoints for `connect`, finally filling the connected three-cell line with gray. The Code Golf solution is an obfuscated arithmetic form of the same count-to-line mapping.
