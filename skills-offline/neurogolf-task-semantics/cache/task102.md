# task102 Semantics

## Sources

- Current champion builder: `solutions_py/task102.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task102.json`
- ARC-GEN task id: `44d8ac46`
- ARC-DSL task id: `44d8ac46`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_44d8ac46.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_44d8ac46.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task102.py`

## Pattern

The grid is a 12x12 canvas containing two or three non-overlapping gray framed
rectangles, sometimes with smaller gray framed rectangles or black junk blocks
inside them. All frame pixels are gray (`5`). Every enclosed interior starts as
black (`0`) in the input. In the output, only enclosed black regions whose
bounding box is square are recolored red (`2`). Enclosed black regions whose
bounding box is not square remain black. The outside background remains black,
even though its bounding box may be square, because it touches the canvas border
and is not enclosed by a gray frame.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, k=7, l=1, a=1: -k * g or p([[(a := [[a | c | (l := (l << 7)), 4228224 >> c % 127 & 2][k < 1], 5][c == 5]) for c in r] for r in zip(*g[::-1])], k - 1, 0)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN always uses a square `12x12` canvas. It first chooses `num_boxes` as
2 or 3 (`min(3, randint(2, 5))`). Base frames have width and height in `[4, 6]`
and are placed without overlap with a one-cell margin. For each base frame, the
generator may leave it alone, mark square-frame interiors as red in the output,
add a smaller red square frame inside a square base frame, or add black junk with
one side of length 2 and the other side length 2 or 3. The input always draws
all frame areas as gray and all interiors/junk/background as black. The output
keeps gray frames unchanged, fills square enclosed black interiors red, and
keeps non-square enclosed interiors black.

## Reference Notes

The ARC-DSL solution computes objects, takes their `delta` interior/background regions, filters those regions with `square`, and fills the selected cells with color `TWO`. This confirms the operation is square-region detection on enclosed black deltas, not rectangle-outline recoloring. The Code Golf 2025 solution is a compact repeated-rotation bit trick, but it follows the same invariant: scan for black enclosed square interiors and change those black cells to red.
