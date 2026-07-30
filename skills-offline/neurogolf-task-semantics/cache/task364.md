# task364 Semantics

## Sources

- Current champion builder: `solutions_py/task364.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task364.json`
- ARC-GEN task id: `e509e548`
- ARC-DSL task id: `e509e548`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e509e548.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e509e548.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task364.py`

## Pattern

The input is a rectangular grid, generally around 10-20 rows and 8-22 columns, containing several separated green (`3`) glyphs on black background. Each glyph is generated as one of three thin sprites inside a 3-5 by 3-5 box:

- `el`: an L-like shape.
- `you`: a U-like shape.
- `aitch`: an H-like shape.

The output preserves the grid size and glyph locations, but recolors each green glyph by its shape class:

- L-like glyphs become blue (`1`).
- H-like glyphs become red (`2`).
- U-like glyphs become pink/magenta (`6`).
- Background remains black (`0`).

## Readable Python Solver

```python
from collections import deque


def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    seen = [[False] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] != 3:
                continue

            q = deque([(r, c)])
            seen[r][c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 3:
                        seen[nr][nc] = True
                        q.append((nr, nc))

            rs = [rr for rr, _ in cells]
            cs = [cc for _, cc in cells]
            r0, r1 = min(rs), max(rs)
            c0, c1 = min(cs), max(cs)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            count = len(cells)

            # L sprites have exactly height + width - 1 cells.
            if count == height + width - 1:
                color = 1
            else:
                inner = [
                    (rr, cc)
                    for rr, cc in cells
                    if r0 < rr < r1 and c0 < cc < c1
                ]
                # H sprites contain green pixels in the trimmed interior;
                # U sprites do not under the generator/reference solver.
                color = 2 if inner else 6

            for rr, cc in cells:
                out[rr][cc] = color

    return out
```

## Generator Constraints

- The generated input height is `10..20`.
- The generated input width is `height + randint(-2, 2)`.
- There are `1 + max(height, width) // 3` glyph boxes.
- Each glyph box has width and height from `3..5`.
- Boxes are non-overlapping with a one-cell margin.
- Every input glyph pixel is green (`3`), regardless of its hidden sprite class.
- Output color depends only on sprite class: `el -> 1`, `you -> 6`, `aitch -> 2`.

## Reference Notes

The ARC-DSL solver extracts green objects. It classifies L-shapes by `size == height + width - 1`, uses a trimmed-subgrid interior test to identify H-shapes, recolors all green cells to `6` by default, then fills H-shapes with `2` and L-shapes with `1`.

The Code Golf solution recursively rotates/transposes the grid and uses local neighborhood logic to identify bends and glyph classes. It is compact but less direct than the object-classification rule above.
