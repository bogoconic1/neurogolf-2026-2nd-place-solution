# task076 Semantics

## Sources

- Current champion builder: `solutions_py/task076.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task076.json`
- ARC-GEN task id: `36d67576`
- ARC-DSL task id: `36d67576`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_36d67576.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_36d67576.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task076.py`

## Pattern

The input contains 3 or 4 separated copies of the same small sprite under
rotation/reflection. One copy is the reference copy: it contains yellow `4` and
red `2` anchor cells plus the missing payload colors blue `1` and green `3`.
Every other copy contains only the yellow/red anchor cells. The output is the
same canvas after painting the blue/green payload cells into every matching
copy in the corresponding transformed orientation.

The reference copy is the nonzero object with the most distinct colors. Its
yellow/red cells identify the anchor shape; its blue/green cells are the payload
to transfer. For each symmetry of the reference sprite, find every placement
whose yellow/red anchor cells match the input, then paint the entire transformed
reference sprite at that placement. Existing anchor cells are preserved because
the painted reference has the same yellow/red values there.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(r):
    a = r
    for n in range(len(r)):
        for f in range(len(r[0])):
            if a[n][f] == 1:
                z = {(f, n)}
    for n in r:
        z = {(f + l, e + n) for l, n in z for f in range(-1, 2) for e in range(-1, 2) if e + n in range(len(r)) != f + l in range(len(r[0])) != 0 < r[e + n][f + l]}
    for n in (1, 1, -1) * 4:
        r = [z for *z, in zip(*r[::n])]
        for e in range(-13, 13):
            for f in range(-13, 13):
                if all((e + n in range(len(r)) != f + l in range(len(r[0])) != a[n][l] in (1, 3, r[e + n][f + l]) for l, n in z)):
                    for l, n in z:
                        r[e + n][f + l] = a[n][l]
    return r


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN uses width and height from 13 to 15. The base sprite starts from a
diagonally connected set of five yellow cells in one of three shapes: `3x3`,
`4x2`, or `5x1`, then adds one red cell, one or two blue cells, and one to
three green cells adjacent to yellow cells without collisions. The red cell is
not placed in the center row or center column of the base sprite, which helps
orientation disambiguation.

There are 3 or 4 copies, placed with at least a one-cell gap. Each copy uses a
random orientation branch. The first copy is the complete reference and is
visible with all colors. Later copies expose only yellow and red anchor cells in
the input; their blue and green payload cells appear only in the output. Output
dimensions match input dimensions.

## Reference Notes

ARC-DSL selects the object with the most colors as the reference, filters its yellow/red anchor cells, generates mirrored/rotated variants, searches those anchor variants as occurrences in the input, then paints shifted full reference variants. The Code Golf solution follows the same idea: expand from the blue payload to recover the reference sprite, rotate/mirror the grid through several orientations, search for anchor-compatible placements, and write missing cells.

The generator names the orientation argument `megarotate`, but the actual branches include rotations and a mirror-like branch. The DSL solver is more general and covers the dihedral symmetries, so the cache treats all eight symmetries as valid search variants.
