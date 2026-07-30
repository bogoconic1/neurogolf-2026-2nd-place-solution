# task115 Semantics

## Sources

- Current champion builder: `solutions_py/task115.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task115.json`
- ARC-GEN task id: `4be741c5`
- ARC-DSL task id: `4be741c5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4be741c5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4be741c5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task115.py`

## Pattern

The input is a striped grid made from 3 or 4 colors. In the landscape orientation, each row is a run-length encoding of the same color order across columns, with noisy/wandering boundaries between adjacent color bands. In the portrait orientation, the same structure is transposed. The output is just the ordered list of distinct stripe colors: a 1xN row for landscape inputs or an Nx1 column for portrait inputs.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, F={}.fromkeys: [*F(zip(*zip(*map(F, g))))]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Number of stripe colors is 3 or 4.
- Colors are nonzero random colors.
- Landscape width is made from color-band thicknesses and gap widths; height is 8..16.
- Gap boundaries wander by at most two cells per row, but the color order is constant.
- `xpose` optionally transposes both input and output, turning the row color list into a column list.
- Padded NeuroGolf rows/columns beyond the ARC grid are all-zero.

## Reference Notes

- ARC-DSL tests `portrait`, conditionally diagonal-mirrors to a landscape view, crops the first row, dedupes adjacent equal colors, then mirrors back if needed.
- Code Golf uses nested `zip`/`dict.fromkeys` to transpose/dedupe/reorient compactly.
