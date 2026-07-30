# task245 Semantics

## Sources

- Current champion builder: `solutions_py/task245.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task245.json`
- ARC-GEN task id: `a1570a43`
- ARC-DSL task id: `a1570a43`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a1570a43.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a1570a43.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task245.py`

## Pattern

The input contains a sparse red (`2`) sprite and a green (`3`) square-frame marker represented by only the four corners of a 7x7 box. The red sprite has the same shape as the target interior sprite, but it is displaced away from its correct location. Move the entire red sprite so that its upper-left occupied cell lands exactly one row and one column inside the upper-left green corner. Keep the green corner pixels fixed, turn the old red locations back to background (`0`), and preserve the grid size. There is one red sprite and one green corner frame; no tie-breaking between multiple objects is needed.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]

    red = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 2]
    green = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 3]

    red_r0 = min(r for r, _ in red)
    red_c0 = min(c for _, c in red)
    green_r0 = min(r for r, _ in green)
    green_c0 = min(c for _, c in green)

    dr = green_r0 + 1 - red_r0
    dc = green_c0 + 1 - red_c0

    for r, c in red:
        out[r][c] = 0
    for r, c in red:
        out[r + dr][c + dc] = 2
    return out
```

## Generator Constraints

ARC-GEN samples have width and height from 7 to 10. The green frame has fixed size 7 and consists only of its four corner pixels, placed at `(brow,bcol)`, `(brow,bcol+6)`, `(brow+6,bcol)`, and `(brow+6,bcol+6)`. The red object is a Conway sprite generated inside the 5x5 interior of that box, then shifted in the input. Random generated examples shift it either left by at least one column or upward by at least one row; the generator source notes diagonal shifts are not random-generated, but one fixed training case uses `roff=-1, coff=-1`, so a correct solver should support both axes at once. The shifted red sprite remains within the grid because the offset magnitude is bounded by `brow + 1` or `bcol + 1`.

## Reference Notes

The ARC-DSL solver takes the upper-left corners of the red set and green set, computes `green_ul - red_ul`, increments both coordinates, and moves the red object by that vector. This confirms that the target location is one cell inside the green upper-left corner, not merely aligned with the corner itself. The Code Golf solution implements the same effect by repeated rotation/shift logic until the red object is in the framed interior. The only ambiguity is generator coverage: random generation lacks diagonal shifts, but the validated training examples include one, and the ARC-DSL rule handles it naturally.
