# task161 Semantics

## Sources

- Current champion builder: `solutions_py/task161.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task161.json`
- ARC-GEN task id: `6cdd2623`
- ARC-DSL task id: `6cdd2623`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6cdd2623.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6cdd2623.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task161.py`

## Pattern

The input is a rectangular grid with random single-pixel noise in two colors and a rarer third marker color placed as matching endpoint pairs on opposite borders. A horizontal laser is encoded by two marker pixels at `(row, 0)` and `(row, width-1)`. A vertical laser is encoded by two marker pixels at `(0, col)` and `(height-1, col)`. The output discards all noise and draws only the complete horizontal and vertical marker-color lines between those paired endpoints on a black background.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [[sum({min((S := sum(g, [])), key=S.count)} & {a, b}) for b in g[0]] for *r, a in g]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

Width is 15..25 and height is 10..20. Random noise pixels cover about 10% of cells and use only two colors from a three-color palette. The third color is the marker/laser color. At least two laser endpoint pairs are generated in total, split arbitrarily between horizontal rows and vertical columns. Horizontal endpoint rows are chosen from `1..height-2`; vertical endpoint columns are chosen from `1..width-2`, so generated lasers do not lie on the outer corners. The marker color appears only in paired border endpoints in the input, while the output contains full marker-color rows and columns on black.

## Reference Notes

The ARC-DSL solver chooses the least frequent color, collects all cells of that color, connects every pair, filters connections that are horizontal or vertical and have interior cells inside the bounding box, clears all foreground noise, and fills those line cells with the marker color. The Code Golf solution expresses the same idea tersely: choose the least frequent nonzero color and keep rows/columns whose two border endpoints have that color.
