# task218 Semantics

## Sources

- Current champion builder: `solutions_py/task218.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task218.json`
- ARC-GEN task id: `90c28cc7`
- ARC-DSL task id: `90c28cc7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_90c28cc7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_90c28cc7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task218.py`

## Pattern

The input is a 21x21 black grid containing one rectangular quilt made from a
small `tall x wide` grid of solid-color rectangular patches. `wide` and `tall`
are each 2 or 3. Every quilt column has a variable width, every quilt row has a
variable height, and the whole quilt can be offset from the top-left. The output
is the compact patch-color grid: one cell per solid rectangle, preserving the
row/column order of the quilt.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, *h: [*{r: 0 for r in zip(*(h or p(*g))) if any(r)}]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN uses a fixed 21x21 input grid. It samples `wide` and `tall` from 2..3.
Patch colors are random nonzero colors subject to two structural constraints:
no two patch rows are identical, no two patch columns are identical, and no cell
has more than one same-color orthogonal neighbor among its patch-grid neighbors
(the generator rejects L-shaped same-color ambiguity). Column widths and row
heights are each 3..12, with total quilt width/height strictly less than 21.
The quilt row and column offsets are arbitrary values that keep the quilt inside
the 21x21 grid.

## Reference Notes

ARC-DSL finds the colored object, crops its bounding subgrid, deduplicates consecutive equal rows, rotates, deduplicates consecutive equal columns, and rotates back. The Code Golf solution is the same idea recursively using `zip`: remove consecutive duplicate rows, then transpose and repeat to remove duplicate columns. ARC-GEN confirms that the output is exactly the compact color grid of patch labels.
