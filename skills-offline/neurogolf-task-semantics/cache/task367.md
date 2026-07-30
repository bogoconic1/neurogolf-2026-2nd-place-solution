# task367 Semantics

## Sources

- Current champion builder: `solutions_py/task367.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task367.json`
- ARC-GEN task id: `e73095fd`
- ARC-DSL task id: `e73095fd`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e73095fd.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e73095fd.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task367.py`

## Pattern

The input is a 10 to 20 row by 10 to 20 column grid containing gray (`5`) hollow rectangular boxes on a black (`0`) background. Boxes have gray borders and black interiors. Some boxes may be clipped by one column at the left or right edge, so a true interior can touch the canvas edge. Gray one-cell horizontal or vertical connector lines may link boxes to each other or to the canvas edge. The output preserves all gray cells and all exterior/background black cells, and recolors only the true black box interiors to yellow (`4`). The output size is unchanged.

Operationally, the target cells are the solid 4-connected black components that fill their own bounding rectangle and are bordered, on every in-bounds visible side, by gray box wall cells. Background black regions and black pockets created by connector lines must remain black even when near gray.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, k=94: ~k * g or p([(g := [c * ((r != [5] * 3) > c % -4 < r[2]) | (k ^ 95 in l) * 4 for *r, c in zip([4] * 2 + g, g, [4] + l, l)]) for *l, in zip(*g[::k % 3 - 1 | 1])], k - 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Width and height are independently sampled from 10 through 20.
- There are 2 to 4 hollow boxes. Box widths and heights are 3 through 7.
- Boxes are non-overlapping according to their possibly clipped rectangles. Horizontal placement may start at `-1` or extend one column past the right edge; vertical placement is fully in bounds.
- Box interiors are at least 1 cell wide and 1 cell tall and are drawn black over the initial gray filled rectangle.
- Connector lines are one-cell wide, horizontal or vertical, gray, and are added only when a three-cell-wide clearance test prevents them from overlapping existing box bitmap cells. Lines can connect box-to-box or box-to-edge.
- Lines and boxes preserve gray in the output. Only box interior black cells become yellow.

## Reference Notes

The ARC-DSL solution finds black objects, keeps those whose cells equal their rectangular backdrop, filters by a gray-neighborhood test around the expanded box corners, and fills the surviving object cells with yellow. That confirms that the semantic object is a rectangular black component, not an arbitrary flood fill from gray.

The Code Golf 2025 entry is a highly compressed iterative expression. It repeatedly scans/transposes and uses local tests involving gray triples and black cells to converge on the same set of interior cells. It is useful evidence that local frame propagation is viable, but it does not expose a simpler global invariant than the rectangular black-component rule.
