# task042 Semantics

## Sources

- Current champion builder: `solutions_py/task042.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task042.json`
- ARC-GEN task id: `22233c11`
- ARC-DSL task id: `22233c11`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_22233c11.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_22233c11.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task042.py`

## Pattern

The visible task grid is 10x10. The input contains one or two green (`3`) diagonal pairs of solid `M x M` blocks, where `M` is 1, 2, or 3 and is shared by all objects in the grid. Each object has two green blocks touching along one of the two diagonal directions. The output preserves the green blocks and adds cyan (`8`) `M x M` blocks one step beyond the pair along the same diagonal line, on the two outer sides of the green pair. Background remains black (`0`).

For an unflipped object, the input green blocks are at `(row, col)` and `(row+M, col+M)`, and the cyan blocks are placed at `(row-M, col+2M)` and `(row+2M, col-M)`. For a flipped object, the input green blocks are at `(row, col+M)` and `(row+M, col)`, and the cyan blocks are at `(row-M, col-M)` and `(row+2M, col+2M)`.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g, n=23: -n * g or [*zip(*eval(re.sub('(?=0.*0(, 3){%d}, 0)(?=.{%d}3)' * 2 % ((x := (n % 5)), x * 38, x, x * 67), '8|', str(p(g, n - 1))))[::-1])]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN always uses square 10x10 grids. It samples `M` uniformly from 1, 2, or 3, then places at least one object with top row and left reference column in the interior range `1..size-2*M-1`. A second object may be placed to the right if there is horizontal room (`col + 6*M < size`), with its reference column at least `3*M` beyond the first. Each object independently chooses one of two diagonal orientations (`flip` 0 or 1). Only green appears in the input; cyan appears only in the output. The placement margins guarantee the added cyan blocks stay inside the 10x10 grid.

## Reference Notes

The ARC-DSL solution extracts foreground objects, mirrors/upscales their shape, shifts it by a vector derived from half the object shape, removes the original frontier cells, and fills the resulting added cells with color 8. The Code Golf solution uses repeated regex/string rotations to add cyan around green diagonal block pairs; it confirms that the transformation is a size/orientation-dependent stamp rather than a color-selection task.
