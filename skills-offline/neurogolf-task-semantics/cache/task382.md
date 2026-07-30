# task382 Semantics

## Sources

- Current champion builder: `solutions_py/task382.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task382.json`
- ARC-GEN task id: `f15e1fac`
- ARC-DSL task id: `f15e1fac`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f15e1fac.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f15e1fac.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task382.py`

## Pattern

The input is a black grid containing red (`2`) marker pixels along one edge and cyan (`8`) marker pixels along a perpendicular edge.  The grid may be horizontally flipped and then rotated/reflected by the generator's gravity transform.  In canonical orientation, the red markers are on the left edge and the cyan markers are on the top edge.  The output keeps all original markers and extends the cyan markers downward.  As the cyan rays pass each red marker row, all subsequent cyan positions shift one column to the right; cyan cells shifted beyond the grid width are omitted.  The output has the same dimensions as the input and is transformed back to the original orientation.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, n=3, u=[]: -n * g or p([*zip(*[[*map(max, r * str(g[1:]).count('8') + (u := ((r[0] == 2) * [0] + u[r[-1] == 2:] + r)), r)] for *r, in g][::-1])], n - 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

Generator dimensions are width and height 10..20.  Red markers are placed along one side at row gaps of 4..7, so there are at least one and usually a few red rows.  Cyan markers are placed along the perpendicular side at column gaps of 2..4.  A random horizontal flip and one of four gravity/orientation transforms are applied after the canonical construction.  Only colors 2 and 8 are nonzero.  Cyan rays that shift beyond the valid grid width are clipped.

## Reference Notes

The ARC-DSL solver first canonicalizes orientation with diagonal mirror, vertical mirror, and horizontal mirror branches so red is on the left edge and cyan is on the top edge. It shoots cyan rays downward, splits them into bands between red marker rows, shifts each band right by the number of red markers already encountered, fills cyan, then applies the inverse orientation transforms. The compact Code Golf solution performs repeated rotations and accumulates a shift list from red markers, matching the same gravity/shift interpretation.
