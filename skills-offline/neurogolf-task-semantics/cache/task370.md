# task370 Semantics

## Sources

- Current champion builder: `solutions_py/task370.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task370.json`
- ARC-GEN task id: `e8dc4411`
- ARC-DSL task id: `e8dc4411`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e8dc4411.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e8dc4411.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task370.py`

## Pattern

The input has a nonzero background color, one black (`0`) sprite, and one single pixel in another color. The colored pixel marks the next diagonal copy of the black sprite. The output keeps the original black sprite and background, then repeatedly stamps the same sprite shape in the marker color along that diagonal direction until the shifted copies leave the grid. The grid size is unchanged.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g: exec('s=f"{*zip(*g[::-1]),}";i=s.rfind;d=i("0")-i(j:=min(s,key=i));g[:]=eval(re.sub("(?=(.{%d})+0)\\d"%d,j,s,d%8*d%-(len(g)*3+5)));' * 4) or g


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN uses width and height from 10 to 20. The sprite side length is 3 or 4 in generated samples, while public examples include related hand-fixed lengths. The black sprite is formed from a Conway-style binary sprite and its symmetries, so the black mask is inside a square bounding box and always includes `(0,0)`. A single marker pixel of `color` is placed one sprite-size step diagonally from the original black sprite. `flip` and `xpose` can turn the effective direction into any diagonal sign/orientation. The background color is nonzero and excludes the marker color.

## Reference Notes

The ARC-DSL solver identifies the least frequent nonzero color as the marker color, extracts the black object, infers the relative position of the marker color to the black object, multiplies that by the black object shape, and fills repeated shifts of the black object at offsets `1..4`. It has a branch for whether the black shape lies exactly on its main diagonal, which adjusts the inferred step for asymmetric edge cases. The Code Golf 2025 solution implements the same repeated diagonal stamping with string/regex transformations over transposed/reversed grids.
