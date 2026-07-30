# task154 Semantics

## Sources

- Current champion builder: `solutions_py/task154.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task154.json`
- ARC-GEN task id: `6855a6e4`
- ARC-DSL task id: `6855a6e4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6855a6e4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6855a6e4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task154.py`

## Pattern

The input is a `15x15` grid containing a red (`2`) gripper-like rectangular frame and gray (`5`) pixels outside that frame. The frame has full red top and bottom horizontal bars, plus red corner posts one row inside each end. The target output preserves the red frame and moves the gray pixels into the empty interior of the frame. Gray pixels originally above the top red bar are mirrored/flipped down into the upper interior. Gray pixels originally below the bottom red bar are mirrored/flipped up into the lower interior. A random branch may transpose both input and output, so the same rule can appear rotated by 90 degrees.

## Readable Python Solver

```python
def solve(grid):
    # Normalize orientation so the red gripper is portrait: top/bottom red bars.
    def transpose(g):
        return [list(row) for row in zip(*g)]

    g = [row[:] for row in grid]
    red = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == 2]
    rows = [r for r, _ in red]
    cols = [c for _, c in red]
    portrait = (max(rows) - min(rows)) >= (max(cols) - min(cols))
    if not portrait:
        g = transpose(g)
        red = [(r, c) for r, row in enumerate(g) for c, v in enumerate(row) if v == 2]
        rows = [r for r, _ in red]
        cols = [c for _, c in red]

    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)
    out = [row[:] for row in g]

    # Clear gray pixels outside the frame.
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v == 5:
                out[r][c] = 0

    # In generator coordinates, interior target rows are top+2 through bottom-2.
    # Upper outside gray rows are mirrored from above the top bar into upper interior.
    # Lower outside gray rows are mirrored from below the bottom bar into lower interior.
    for r, row in enumerate(g):
        for c, v in enumerate(row):
            if v != 5:
                continue
            if top - 5 <= r <= top - 2:
                rr = top + (top - 2 - r) + 2
            elif bottom + 2 <= r <= bottom + 5:
                rr = bottom - (r - bottom - 2) - 2
            else:
                continue
            out[rr][c] = 5

    return out if portrait else transpose(out)
```

## Generator Constraints

ARC-GEN uses a fixed `15x15` canvas. In the untransposed orientation the red frame top row is `brow=3`, width is `5..7`, height is `8..9`, and left column is `2..5`. The gray pattern uses internal relative rows and columns drawn from a connected random shape over width `wide-2` and height `tall-4`; top and bottom halves are each diagonally connected and centrally aligned. Top-half gray cells are drawn above the top bar; bottom-half cells are drawn below the bottom bar. Output places all gray cells inside the frame at row `brow + r + 2`, column `bcol + c + 1`. The generator may transpose the whole input/output.

## Reference Notes

ARC-DSL normalizes orientation using a red-object portrait test, extracts gray objects, pairs the upper/lower gray pieces by center/flip, mirrors each piece horizontally in normalized orientation, shifts them into the frame interior, covers the old gray pixels, paints the shifted gray pixels, and rotates back if needed. The Code Golf 2025 solution uses a compact regex/transpose trick over the same red-frame geometry.
