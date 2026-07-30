# task293 Semantics

## Sources

- Current champion builder: `solutions_py/task293.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task293.json`
- ARC-GEN task id: `ba97ae07`
- ARC-DSL task id: `ba97ae07`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ba97ae07.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ba97ae07.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task293.py`

## Pattern

The input is a small rectangular grid, padded by the evaluator into the standard `[1,10,30,30]` one-hot tensor. The visible grid contains exactly two solid stripes on a black background: one horizontal stripe and one vertical stripe. Each stripe has thickness 1 or 2. The two stripes cross, and the color visible at the crossing is the stripe that was drawn second.

The output keeps the same stripes and background, but swaps the color at the overlap to the color of the stripe that was underneath in the input. In other words, only the intersection rectangle changes; the top stripe visible color there is replaced by the other stripe color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [[y - x + r[0] or x for x, y in zip(r, g[0])] for r in g]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN chooses width and height independently from 5 through 15. It chooses two non-background colors, a vertical stripe thickness in `{1, 2}`, and a horizontal stripe thickness in `{1, 2}`. The vertical stripe offset is at least 1 and leaves at least one column to its right; the horizontal stripe offset is at least 1 and leaves at least one row below it. A `flip` bit controls draw order: if `flip == 0`, the horizontal stripe is drawn first and the vertical stripe is on top in the input; if `flip == 1`, the vertical stripe is drawn first and the horizontal stripe is on top. The output redraws the same two stripes in the opposite order, so only the overlap changes color.

Important edge cases:

- visible grid sizes vary from 5x5 to 15x15, but the NeuroGolf tensor is still padded to 30x30
- stripe thickness can be 1 or 2 in either direction
- either stripe color may be numerically smaller or larger; color order is not meaningful
- the overlap always exists and is a full rectangle of size `horizontal_thickness x vertical_thickness`

## Reference Notes

ARC-DSL finds connected non-background objects with diagonal connectivity enabled, takes the most common non-background color, finds all cells of that color, takes their bounding box, and fills that box with the same most-common color. The ARC-GEN source is clearer: it explicitly draws the two stripes in one order for input and the reverse order for output.

The Code Golf 2025 solution uses the first row as a background reference and computes the color difference `y - x + r[0]` elementwise. This works because the non-overlap stripe arms reveal which color is absent at the crossing.
