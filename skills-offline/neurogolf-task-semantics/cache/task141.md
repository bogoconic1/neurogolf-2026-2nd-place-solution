# task141 Semantics

## Sources

- Current champion builder: `solutions_py/task141.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task141.json`
- ARC-GEN task id: `623ea044`
- ARC-DSL task id: `623ea044`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_623ea044.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_623ea044.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task141.py`

## Pattern

The input is an odd-size square grid containing exactly one non-background colored pixel in the interior. The output keeps that pixel color and fills both diagonals passing through it, clipped to the square grid. In coordinate form, if the colored pixel is at row r0 and column c0 with color k, every in-bounds cell (r,c) with r + c = r0 + c0 or r - c = r0 - c0 becomes k; all other cells remain background 0.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, i=1: [[*map(max, r[:(a := abs(g.index((m := max(g))) + (i := (i - 1))))] + m, m[a:] + r, r)] for r in g]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN chooses an odd square size from 5 through 21 inclusive. The seed row and column are both interior, from 1 to size - 2. The seed color is a nonzero ARC color chosen by random_color. There is exactly one foreground object, a single cell. The output square has the same size as the input square. The NeuroGolf tensor is still padded to the standard [1,10,30,30] contract, so masks must not paint outside the true square.

## Reference Notes

ARC-DSL finds the single object, takes its center and color, shoots rays in the two diagonal direction pairs, combines them, and fills the input with the seed color on those ray cells. The Code Golf solution is a compact row-wise diagonal fill. The references agree that there is no tie-breaking, no multiple objects, and no variable output size beyond the input square size.
