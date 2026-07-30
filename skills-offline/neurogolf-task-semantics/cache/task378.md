# task378 Semantics

## Sources

- Current champion builder: `solutions_py/task378.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task378.json`
- ARC-GEN task id: `ec883f72`
- ARC-DSL task id: `ec883f72`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ec883f72.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ec883f72.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task378.py`

## Pattern

The input is a square black grid containing a two-color nested rectangular figure. The larger nonzero object is an outer frame color. Inside it is a one-cell black moat and then a filled inner rectangle in the other nonzero color. The figure may be clipped by the grid boundary.

The output preserves the input and adds four diagonal rays in the inner-rectangle color. The rays extend outward from the four corners of the outer frame: up-left from the upper-left frame corner, up-right from the upper-right frame corner, down-left from the lower-left frame corner, and down-right from the lower-right frame corner. Only black cells are filled; existing colored frame and inner cells stay unchanged.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g: exec(f"""g[::-1]=zip(*eval(re.sub(r"(?=(.{{{(x := (len(g) * 3)) + 5}}})+([^0]), \\2{'..' * x}\\2, 0, (.))0",r"\\3",str(g))));""" * 4) or g


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Grid is square with `size` sampled from 6 through 12.
- Random generator samples inner rectangle width and height from 2 through 5, but validation examples include a 1x1 inner rectangle, so optimizers should not rely on a minimum of 2.
- The inner rectangle top-left `row`, `col` can lie at any coordinate `0..size-1`, so the nested frame and rays can be clipped by the grid boundary.
- Exactly two nonzero colors are used. The outer frame uses `colors[1]`; the inner rectangle and output rays use `colors[0]`.
- The input has the outer frame, a black moat, and the inner rectangle. The output adds only the diagonal rays.
- The generator retries until the output differs from the input, so at least one ray cell is visible.

## Reference Notes

ARC-GEN explicitly draws the outer frame, then overwrites a smaller middle ring with black, then draws the inner filled rectangle. The antenna/ray pixels are drawn only in the output and use the inner color. ARC-DSL identifies the largest nonzero object, takes its color as the frame color, selects the other nonzero color, shoots four diagonal rays from the frame object corners, and underfills the original grid with the ray color. The Code Golf solution confirms a rotation/regex style trick, but ARC-GEN and ARC-DSL are clearer for graph design.

Important optimization caveat: do not identify the frame color by raw pixel count. In clipped cases the visible frame can be an L-shape with fewer cells than the filled inner rectangle; use object/bounding-box geometry or another frame-crossing signal instead.

There is no semantic disagreement between references. The important detail is `underfill`: rays must not overwrite the existing frame or inner rectangle.
