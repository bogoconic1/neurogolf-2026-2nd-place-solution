# task300 Semantics

## Sources

- Current champion builder: `solutions_py/task300.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task300.json`
- ARC-GEN task id: `be94b721`
- ARC-DSL task id: `be94b721`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_be94b721.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_be94b721.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task300.py`

## Pattern

The input contains 3 or 4 separated colored sprites on a rectangular grid. Each sprite has a distinct color. Sprite sizes are sampled from `3..9` and sorted descending before placement, so sprite index `0` is the largest. The output is the tight bounding-box crop of that largest sprite, preserving its color and shape and moving it to the top-left of the output grid.

Equivalently, find the non-background color with the highest cell count, find the rows and columns containing that color, and return the subgrid induced by those rows and columns.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter

    counts = Counter(v for row in grid for v in row if v != 0)
    color = max(counts, key=counts.get)
    rows = [r for r, row in enumerate(grid) if color in row]
    cols = [c for c in range(len(grid[0])) if any(grid[r][c] == color for r in range(len(grid)))]
    return [[grid[r][c] for c in cols] for r in rows]
```

## Generator Constraints

- Number of sprites is effectively `3` or `4` (`max(3, randint(1,4))`).
- Input height is `5..9`; input width is `9..(20-height)`, so examples fit in a small top-left region of the standard 30x30 tensor.
- Sprite sizes are sampled from `3..9` without replacement and sorted descending. The largest sprite is unique.
- Sprite bounding boxes are derived from size: width is `2` for size `<4`, else `3`; height is `2` for size `<6`, `3` for size `6..8`, and `4` for size `9`.
- Sprites are generated as continuous shapes, placed non-overlapping with one-cell padding, and have unique random colors.
- The output crop size is the largest sprite's bounding box, up to `3x4` or `4x3` depending on generated shape orientation constraints.

## Reference Notes

The ARC-DSL solver finds all objects, selects the one with maximum size, and returns `subgrid` over that object. The Code Golf solution finds the most frequent nonzero color and recursively filters rows and columns to those containing that color, producing the same tight crop.
