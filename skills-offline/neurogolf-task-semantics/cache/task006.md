# task006 Semantics

## Sources

- Current champion builder: `solutions_py/task006.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task006.json`
- ARC-GEN task id: `0520fde7`
- ARC-DSL task id: `0520fde7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_0520fde7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_0520fde7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task006.py`

## Pattern

The input is a fixed 3x7 grid. Columns 0..2 and 4..6 contain blue `1` pixels, separated by a gray `5` vertical bar in column 3. The output is a 3x3 grid. For each row and each column in the left half, output red `2` exactly where the left-half blue pixel and the horizontally mirrored right-half blue pixel are both present; otherwise output black `0`.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [eval('r.pop(0)*r[3]*2,' * 3) for r in g]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN uses width `3` for each half and height `3`, so inputs are always `3x7` and outputs are always `3x3`. It samples random blue pixels on both halves and rejects samples unless each half has at least one blue pixel. The center separator is always gray `5`. The left and mirrored-right intersection is recolored red `2` in the output.

## Reference Notes

The ARC-DSL solver mirrors the input, takes the mirrored left and right halves, mirrors one half back, intersects them with `cellwise(..., ZERO)`, then replaces blue `1` by red `2`. The Code Golf solution computes the same per-row product of left cells and mirrored right cells, multiplied by 2.
