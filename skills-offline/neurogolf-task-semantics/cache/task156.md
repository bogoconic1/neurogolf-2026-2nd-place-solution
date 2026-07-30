# task156 Semantics

## Sources

- Current champion builder: `solutions_py/task156.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task156.json`
- ARC-GEN task id: `694f12f3`
- ARC-DSL task id: `694f12f3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_694f12f3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_694f12f3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task156.py`

## Pattern

The input is a 10x10 grid containing exactly two yellow (`4`) rectangular frames. The output keeps the yellow frames and fills each frame interior. The interior of the smaller yellow rectangle is filled with blue (`1`), and the interior of the larger yellow rectangle is filled with red (`2`). The rectangles are separated vertically, but the whole grid may be vertically flipped by the generator. Background remains black (`0`).

The two rectangles are guaranteed to be different in exactly one dimension: either they have equal height and different widths, or equal width and different heights. The rectangle with smaller area is therefore unambiguous and receives color `1`; the larger rectangle receives color `2`.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = set()
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 4 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    yy, xx = y + dy, x + dx
                    if 0 <= yy < h and 0 <= xx < w and grid[yy][xx] == 4 and (yy, xx) not in seen:
                        seen.add((yy, xx))
                        stack.append((yy, xx))
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            comps.append((min(rs), max(rs), min(cs), max(cs), len(comp)))

    out = [row[:] for row in grid]
    comps.sort(key=lambda box: (box[1] - box[0] + 1) * (box[3] - box[2] + 1))
    for color, (r0, r1, c0, c1, _size) in zip((1, 2), comps):
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN uses a fixed 10x10 square grid. It chooses a horizontal border row between 3 and 5. Two yellow rectangular frames are placed in two vertical bands: one above the border and one below it. Frame widths (`lengths`) are independently 3..7, and frame heights (`talls`) are bounded by the band size. The generator rejects cases where both dimensions differ, where both dimensions are equal, or where the upper/red-small candidate would be larger in either dimension. Thus one dimension matches and the other grows from smaller to larger. Rectangle columns are random valid placements. A vertical flip may be applied to both input and output, so top/bottom position is not a reliable label.

Input foreground color is only yellow (`4`). Output fill colors are `1` for the smaller frame interior and `2` for the larger frame interior. The yellow border cells stay yellow.

## Reference Notes

The ARC-DSL solver finds yellow objects, computes each frame inside-box interior (`backdrop(inbox(object))`), chooses `argmin` and `argmax` by object size, fills the smaller interior with `1`, then fills the larger interior with `2`. The Code Golf 2025 solution uses a compact regex/string trick but follows the same frame-interior fill rule. There is no ambiguity across the references.
