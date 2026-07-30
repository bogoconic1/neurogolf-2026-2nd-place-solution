# task069 Semantics

## Sources

- Current champion builder: `solutions_py/task069.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task069.json`
- ARC-GEN task id: `321b1fc6`
- ARC-DSL task id: `321b1fc6`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_321b1fc6.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_321b1fc6.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task069.py`

## Pattern

The input contains one original multicolor connected sprite and several cyan copies of the same sprite shape. The output removes the original sprite and repaints every cyan copy with the original sprite's colors in the matching normalized positions.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
R = range(320)

def p(g):
    *f, = U = b'%r' % g
    for i in R:
        for j in (P := [j for j in R if U[j] & 503 % f[i] > 48]):
            f[j], f[i + j - P[0]] = (48, U[j])
    return eval(bytes(f))


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Grid size is fixed at `10x10`.
- Source sprite width is `2..4`, height is `2..3`, and the sampled cells are connected.
- The source sprite uses 2 to 4 non-cyan colors and contains more than one color.
- There are four non-overlapping placements with a one-cell gap. Placement 0 is the original multicolor sprite; later placements are cyan copies.
- Output clears the original sprite to black and paints every cyan copy with the source colors.

## Reference Notes

The ARC-DSL solver finds all objects, filters cyan copies, normalizes the single non-cyan object, covers/removes the original, shifts the normalized source to each cyan object's upper-left corner, and paints those shifted cells. The Code Golf solution encodes the same byte-level copy from the original non-cyan pattern to cyan targets.
