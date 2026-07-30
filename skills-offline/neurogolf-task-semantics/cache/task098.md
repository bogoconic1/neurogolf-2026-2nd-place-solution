# task098 Semantics

## Sources

- Current champion builder: `solutions_py/task098.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task098.json`
- ARC-GEN task id: `4347f46a`
- ARC-DSL task id: `4347f46a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4347f46a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4347f46a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task098.py`

## Pattern

The input is a black grid containing one or more non-overlapping solid colored rectangles. Rectangles may use different nonzero colors. The output keeps each rectangle's one-cell border in its original color and changes every strict interior cell of every rectangle to black. The grid size is unchanged.

A colored cell is an interior cell exactly when its four orthogonal neighbors are also colored cells of the same rectangle/color. Border cells have at least one orthogonal neighbor outside the rectangle and are preserved.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 0:
                out[r][c] = 0
                continue
            interior = True
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr, c + dc
                if not (0 <= rr < h and 0 <= cc < w and grid[rr][cc] == color):
                    interior = False
                    break
            out[r][c] = 0 if interior else color
    return out
```

## Generator Constraints

Height is sampled in `[8, 20]`; width is height plus `-2..2`. The number of boxes is `max(1, (width + height)//5 - 2)`. Each rectangle has width and height in `[3, 8]`, is placed at least one cell away from the outer grid boundary, and rectangles are generated until they do not overlap with one-cell separation. Each rectangle receives a random nonzero color. Inputs are solid filled rectangles on black; outputs hollow each rectangle by blackening rows `row+1..row+tall-2` and columns `col+1..col+wide-2`.

## Reference Notes

The ARC-DSL solver finds objects with 4-connectivity disabled? Its `objects(I, T, F, T)` call groups same-colored rectangular objects, computes `box(object) - toindices(object)` style interior/background fill targets, then fills those targets with zero. The Code Golf solution recursively compares a cell with its four direct neighbors; it preserves borders and removes cells whose left/right/up/down context confirms they are internal to a solid block.
