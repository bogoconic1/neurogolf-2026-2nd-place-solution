# task138 Semantics

## Sources

- Current champion builder: `solutions_py/task138.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task138.json`
- ARC-GEN task id: `5daaa586`
- ARC-DSL task id: `5daaa586`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5daaa586.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5daaa586.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task138.py`

## Pattern

The input contains a rectangular frame made from four colored border lines: left, right, top, and bottom. The four line colors are distinct. Additional isolated pixels of one of those four colors are sprinkled throughout the full grid.

The output is the crop of the frame, including the border. Inside that crop, keep the frame colors and draw straight rays from every sprinkled pixel of the selected color that lies inside the frame. The ray direction is determined by which frame side has the selected color: left-color pixels draw left, right-color pixels draw right, top-color pixels draw upward, and bottom-color pixels draw downward. The ray continues until it reaches the corresponding frame border. Sprinkled pixels outside the frame do not affect the output crop except when their coordinate projects inside the crop as a point before ray clipping in the generator's bounded draw helper.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, k=35: -k * g or p([[(g := [e, g][l[0] == g > e < 1 < 9 > k]) for e in l[::-1]] for *l, in zip(*g[0 in g[0]:])], k - 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Width is `10..25`; height is width plus `-1..1`.
- Frame coordinates are well inset: `left` is in the first third, `right` in the last third, `top` in the first third, and `bottom` in the last third.
- Four distinct frame colors are sampled; the draw color is one of those four colors.
- Sparse pixels are random non-neighboring positions at about 10% density. All sparse pixels use the same draw color.
- Frame lines are drawn after sparse pixels, so frame colors overwrite any sparse pixel on a border.
- Output size is `(down-up+1) x (right-left+1)`, then NeuroGolf pads to `[1,10,30,30]`.

## Reference Notes

The ARC-DSL solver finds the interior-bordering black component, takes its outbox, crops that subgrid, finds the largest foreground object inside the crop, uses its color as the draw color, connects every pair of its pixels, then selects vertical or horizontal line segments depending on which orientation has more segments. This matches drawing the color-coded rays through the crop and frame.

The Code Golf solution is a compact iterative rotate/fill routine. It agrees with the generator behavior: crop around the frame and propagate the selected color along one axis until blocked by the border.
