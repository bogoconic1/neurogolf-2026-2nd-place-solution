# task153 Semantics

## Sources

- Current champion builder: `solutions_py/task153.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task153.json`
- ARC-GEN task id: `681b3aeb`
- ARC-DSL task id: `681b3aeb`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_681b3aeb.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_681b3aeb.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task153.py`

## Pattern

The input is a `10x10` black-background grid containing two separated colored partial `3x3` stamps. The stamps use two different colors and are placed with top-left offsets at least three rows and columns apart, so their bounding boxes do not overlap. For every relative cell in a `3x3` pattern, exactly one of the two stamps contains its color. The output is the reconstructed `3x3` pattern: cells belonging to the first stamp take the first color, and cells belonging to the second stamp take the second color.

Equivalently, choose the larger of the two colored objects, normalize it to its own `3x3` bounding box, create a `3x3` canvas filled with the smaller object's color, and paint the larger object's normalized cells onto that canvas in the larger object's color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
T = (0, 1, 2)
p = lambda g: max((all(sum((G := [[g[x + i % 7][y + i % 8] ^ g[x - i % 9][y - i % 11] for y in T] for x in T]), G)) * G for i in range(5544)))


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN samples two top-left positions in a `10x10` grid, normally from `0..7` with row and column separation greater than two. Validation examples include a negative row for one object, but the emitted visible cells still lie in-grid. The binary `3x3` pattern is generated from a connected creature of size 3 through 6; both the selected cells and their complement must be diagonally connected. The pattern cannot be split evenly by an all-one row or all-one column, avoiding ambiguous simple splits. The two object colors are random distinct ARC colors.

## Reference Notes

ARC-DSL rotates the grid, extracts colored objects, chooses `argmax(size)` and `argmin(size)`, fills a `3x3` canvas with the smaller object's color, paints the normalized larger object, then rotates back. The Code Golf solution brute-forces offsets and XOR-like differences to recover the same merged 3x3 pattern.
