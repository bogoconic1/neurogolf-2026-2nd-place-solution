# task178 Semantics

## Sources

- Current champion builder: `solutions_py/task178.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task178.json`
- ARC-GEN task id: `746b3537`
- ARC-DSL task id: `746b3537`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_746b3537.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_746b3537.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task178.py`

## Pattern

The visible input is a 1D sequence of solid-color stripes, either horizontal or
vertical. Adjacent stripes always have different colors, but a color may appear
again after another stripe. Each stripe has thickness 1, 2, or 3 in the varying
axis and spans the full width/thickness of the other axis. The output is the
deduplicated run-color list in the same orientation: for horizontal stripes, a
single-column grid whose rows are the stripe colors from top to bottom; for
vertical stripes, a single-row grid whose columns are the stripe colors from
left to right.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: g * -1 * -1 or [p((g := r)) for r in g if g != r]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

The generator chooses width 1..5, number of color runs 3..5, and per-run
thickness 1..3. Colors are ARC colors 1..9; only the immediately previous color
is excluded, so non-adjacent repeated colors are legal. The untransposed input
height is the sum of the thicknesses, at most 15, and width is at most 5. With
transpose enabled, height and width swap and the output is transposed from a
`len(colors) x 1` column into a `1 x len(colors)` row. Background black is not
used inside the visible generated grid, only as NeuroGolf padding.

## Reference Notes

ARC-DSL checks whether the first object has size 1 to decide whether to diagonal-mirror/transpose first, then orders objects left-to-right, extracts their colors, repeats each color once, and mirrors back. This is equivalent to deduplicating adjacent stripe runs along the varying axis. The Code Golf solution recursively removes adjacent duplicates from the top-level list, which matches the same run-length color extraction for generated stripe grids. There is no conflicting reference behavior.
