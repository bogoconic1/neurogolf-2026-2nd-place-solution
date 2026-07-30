# task234 Semantics

## Sources

- Current champion builder: `solutions_py/task234.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task234.json`
- ARC-GEN task id: `98cf29f8`
- ARC-DSL task id: `98cf29f8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_98cf29f8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_98cf29f8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task234.py`

## Pattern

The input contains two colored rectangular bodies separated along one axis, with a one-cell-wide connector/tongue in the second body color running through the gap. One foreground component is a plain solid rectangle. The other foreground component is the second rectangle plus its thin connector. The output removes the connector and translates the second rectangle along the connector axis until it touches the first rectangle. The first rectangle remains fixed.

The generator may vertically flip and/or transpose both input and output, so the visible relation may be top/bottom, bottom/top, left/right, or right/left. The operation is orientation-invariant: close the gap along the connector direction while preserving each rectangle's color, size, and orthogonal coordinate.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [(g := [*zip(*[r for r in g[:1] * 99 + g if ~-r.count(max(max(g, key=any)))][:~len(g):-1])]) for _ in g][3]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Input width is between 12 and 20; height is between 15 and 20 before optional transpose.
- There are two foreground colors chosen randomly.
- Body widths are `(7..8, 3..5)` and body heights are `(3..8, 3..4)` before a possible swap that can make either body larger.
- The second body is horizontally aligned within the first body's span before optional transpose; the tongue column lies inside the second body.
- The vertical gap/tongue length is at least 1, and the two bodies do not overlap.
- Optional vertical flip and optional transpose are applied after construction, so implementations must not assume the connector points downward in final coordinates.

## Reference Notes

- ARC-DSL selects the already solid rectangular component, treats the other component as the tongue/body component, extracts the dense body area from that component, removes the full tongued component, then uses `gravitate` to shift the body box until it touches the solid rectangle.
- The Code Golf solution uses repeated transpose/scan tricks to remove rows/columns through the gap until the body is adjacent, confirming that the task is essentially gap compression along one axis.
- The tongue color is the color of the moved body, not a separate marker color.
