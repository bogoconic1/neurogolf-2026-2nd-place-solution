# task074 Semantics

## Sources

- Current champion builder: `solutions_py/task074.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task074.json`
- ARC-GEN task id: `3631a71a`
- ARC-DSL task id: `3631a71a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3631a71a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3631a71a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task074.py`

## Pattern

The input is a `30x30` grid containing a D4-symmetric colored pattern with some
rectangular cutouts colored maroon (`9`). The output replaces the maroon cutouts
with the true hidden colors implied by the pattern's dihedral symmetries. The
visible colored cells are generated from a triangular seed over folded
coordinates, then reflected across the main diagonal, anti-diagonal, vertical,
and horizontal axes of a virtual `32x32` square and clipped to the `30x30` ARC
grid.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: g[90:] or p(g + [*zip(*map(map, [min] * 30, p(g[:30] + g), g[:2] + g[::-1]))])


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Grid size is fixed at `30x30`; the virtual symmetric construction uses
  `trisize=16`, so mirrored coordinates are based on `31 - coord`.
- Seed colors come from a lower-triangular `16x16` region and exclude maroon.
- The generator places `2..5` maroon rectangular cutouts with widths/heights in
  `[2,8]` before drawing the visible symmetric pattern.
- A generated instance is accepted only if every D4 orbit has at least one
  visible non-maroon in-grid cell, so the hidden color is recoverable.
- Output contains no maroon cutouts; it is the fully restored clipped D4 pattern.

## Reference Notes

ARC-DSL replaces maroon with black, combines the grid with a diagonal mirror via per-cell maximum, crops the inner region, mirrors vertically, extracts restored objects, shifts them back by `(2,2)`, and paints them over the partially filled canvas. This compactly exploits the same D4 orbit structure and the fact that outer virtual rows/columns are clipped away.

The Code Golf 2025 solution repeatedly zips reflected rows/columns and takes per-cell minima/maxima to propagate visible orbit colors into the cutouts.

It maps input cells to output cells when their folded row/column coordinates match under the allowed D4 flips, while suppressing maroon channel `9` through a `non9` channel factor.
