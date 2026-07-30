# task134 Semantics

## Sources

- Current champion builder: `solutions_py/task134.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task134.json`
- ARC-GEN task id: `5ad4f10b`
- ARC-DSL task id: `5ad4f10b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5ad4f10b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5ad4f10b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task134.py`

## Pattern

The input is a 20..30 by 20..30 black grid containing sparse noise pixels in one nonzero color and one magnified 3x3 sprite in a second nonzero color. The magnified sprite is made by expanding selected cells of a 3x3 binary pattern by a scale factor 2..6. The output is the original 3x3 pattern, recolored with the sparse noise color rather than the magnified sprite color. Empty pattern cells are black.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, y=1: [(s := [y * (0 < x != y) for *c, x in zip(*g, r) if {*c} - {0, y}])[::(n := (len(s) // 3))] for r in g if {*r} - {0, y}][::n] * (f'0, {y}, 0' in '%r' % g) or p(g, y + 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Input width and height are independently 20..30.
- Sparse noise pixels have density about 5 percent and use one random nonzero color.
- The magnified sprite uses a different random nonzero color.
- The hidden sprite is a Conway-style 3x3 pattern; only selected 3x3 cells are filled.
- Magnification scale is 2..6. The magnified 3x3 block is placed with at least one-cell top/left margin and fits in the input.
- The output is always 3x3 and uses the sparse noise color on pattern cells.

## Reference Notes

ARC-DSL selects the largest object, crops its bounding box, replaces the least color in that crop with black, replaces the mega color with the sparse color, and downscales by one third of the cropped height. Code Golf iterates possible colors and extracts the 3x3 sprite by sampling every scale-sized block, agreeing with the same largest-object and recolor rule.
