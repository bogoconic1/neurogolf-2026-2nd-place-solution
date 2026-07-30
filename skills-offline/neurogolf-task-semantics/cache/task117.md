# task117 Semantics

## Sources

- Current champion builder: `solutions_py/task117.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task117.json`
- ARC-GEN task id: `4c5c2cf0`
- ARC-DSL task id: `4c5c2cf0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4c5c2cf0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4c5c2cf0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task117.py`

## Pattern

The input is a 12x12 to 15x15 square grid containing a five-cell 3x3 body marker in one color and one diagonally connected leg/blob in a second color. The body marker occupies the four corners and center of a 3x3 square. The leg/blob appears in one diagonal quadrant around the body. The output keeps the input and adds the other three symmetric leg/blob copies, mirrored horizontally, vertically, and both horizontally and vertically around the body center.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, n=-79: n * g or p([*zip(*[g, (h := [*map(max, g, (i := (g * 3)[n % -21::-1]))])][40 in map(str(i + 6 * h).count, str(h))])], n + 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Square size is 12..15.
- The leg/blob is generated from a Conway-style diagonally connected sprite with width and height in a small range derived from the grid size.
- The body top-left offset is chosen so the leg/blob and all mirrored copies fit in the grid.
- The body color and leg color are distinct nonzero colors.
- Optional horizontal and vertical flips move the existing leg/blob to any diagonal quadrant while preserving the same mirror-completion rule.
- NeuroGolf pads the input/output to `[1,10,30,30]`; the active scalar canvas never needs to exceed 15x15.

## Reference Notes

- ARC-GEN explicitly draws one leg/blob in the input and all four symmetric copies in the output, then applies optional flips.
- ARC-DSL finds connected objects, identifies the body by comparing a subgrid with its rotation, mirrors/paints one missing side, then mirrors/paints the other side.
- Code Golf expresses the same repeated mirror-completion compactly with transposition/zip tricks.
