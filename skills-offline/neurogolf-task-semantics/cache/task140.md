# task140 Semantics

## Sources

- Current champion builder: `solutions_py/task140.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task140.json`
- ARC-GEN task id: `6150a2bd`
- ARC-DSL task id: `6150a2bd`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6150a2bd.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6150a2bd.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task140.py`

## Pattern

The input is a fixed 3x3 grid. The output is the same 3x3 grid rotated 180 degrees: out[r][c] = grid[2-r][2-c]. All colors are ordinary ARC digits 0..9, and repeated colors are allowed. The ARC-GEN generator fills six named positions directly and leaves the other three positions as the default background 0; because colors may also include 0, the solver must preserve literal color values rather than treat 0 as empty.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [r[::-1] for r in g[::-1]]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN always emits square grids of size 3. The generated color list has six values in 0..9, with no uniqueness constraint. The populated input cells are (0,0), (0,1), (0,2), (1,0), (1,1), and (2,0); default zero cells may appear at (1,2), (2,1), and (2,2). The expected output places those values at the 180-degree rotated coordinates. The task has no object search, variable dimensions, tie-breaking, or stochastic branch beyond colors.

## Reference Notes

ARC-DSL is exactly rot180(I). The Code Golf solution reverses row order and each row. These agree with ARC-GEN. No ambiguity was found; the only important edge case is that color 0 can be a true payload color, so zero padding or implicit zero values must not be decoded as a real output color except inside the 3x3 rotated region.
