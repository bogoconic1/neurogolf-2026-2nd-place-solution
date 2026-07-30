# task290 Semantics

## Sources

- Current champion builder: `solutions_py/task290.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task290.json`
- ARC-GEN task id: `b94a9452`
- ARC-DSL task id: `b94a9452`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b94a9452.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b94a9452.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task290.py`

## Pattern

The input contains one square cookie-like object on a black background. The object is a square frame of one color with a centered square of another color. The frame thickness and inner square size are each 1 or 2 cells, so the total object size is 3 through 6. The output is the tight crop of that object with the two non-background colors swapped: the frame becomes the inner color and the center becomes the frame color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, *u: [sum({*u}, r * -1) or p(r, *sum(g, r)) for r in g if [r] > g]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN chooses visible width and height from 10 through 15. It chooses frame/center thicknesses in 1..2, making object size thicks[0] + 2 * thicks[1], i.e. 3..6. The object is placed with at least one cell margin from the grid border. Two non-background colors are chosen. The full square is first filled with the frame/cookie color, then the centered thicks[0] by thicks[0] square is filled with the other color. The output is exactly the object crop with those two colors swapped.

## Reference Notes

ARC-DSL extracts the first object, takes its subgrid, finds leastcolor and mostcolor inside the crop, and switches those colors. The Code Golf solution is a terse recursive crop/color-swap implementation.
