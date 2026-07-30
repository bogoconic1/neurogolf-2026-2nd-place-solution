# task308 Semantics

## Sources

- Current champion builder: `solutions_py/task308.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task308.json`
- ARC-GEN task id: `c8cbb738`
- ARC-DSL task id: `c8cbb738`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c8cbb738.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c8cbb738.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task308.py`

## Pattern

The input is a mostly uniform background grid containing 2 or 4 foreground colors. Each foreground color appears as a symmetric set of pixels around its own center. The pixels for a color occupy positions drawn from one perimeter of a hidden square: either a vertical/horizontal cross when the offset row is 0, or the four sign-reflections of one `(row, col)` offset.

Group all non-background pixels by color. For each color, take its bounding box, normalize that color's pixels to the box origin, and center the normalized shape inside an output square. The output square side is the largest foreground span, always one of `3`, `5`, or `7` for generated examples. Paint every centered normalized color group onto a background-colored square. The final NeuroGolf tensor still uses the dense `[1, 10, 30, 30]` output contract, so the small square is represented in the upper-left region and padded with zeros outside the square.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter, defaultdict

    bg = Counter(v for row in grid for v in row).most_common(1)[0][0]

    by_color = defaultdict(list)
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v != bg:
                by_color[v].append((r, c))

    boxes = {}
    n = 0
    for color, pts in by_color.items():
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1
        boxes[color] = (pts, r0, c0, bh, bw)
        n = max(n, bh, bw)

    out = [[bg for _ in range(n)] for _ in range(n)]
    for color, (pts, r0, c0, bh, bw) in boxes.items():
        roff = (n - bh) // 2
        coff = (n - bw) // 2
        for r, c in pts:
            out[roff + r - r0][coff + c - c0] = color
    return out
```

## Generator Constraints

- Hidden half-size is `1`, `2`, or `3`, so the small output side is `3`, `5`, or `7`.
- Input width and height are approximately `5 * halfsize + 3`, each perturbed by `-2..2`; generated foreground centers are at least `halfsize` cells away from the border.
- The background is one random color. Foreground colors are sampled excluding the background.
- The number of foreground elements is 2 or 4, forced to 2 for half-size 1 and to 4 for half-size 3.
- Candidate offsets are sampled from `(i, halfsize)` for `i < halfsize` and `(halfsize, i + 1)` for `i < halfsize`. Offset row zero additionally draws the transposed symmetric points, creating the full cross.
- The generator rejects placements that would overlap existing non-background pixels, so every foreground color can be recovered independently.

## Reference Notes

The ARC-DSL solution uses `mostcolor`, `fgpartition`, `normalize`, and centered `shift`. `fgpartition` groups all cells of the same foreground color together; it is not connected-component partitioning. The Code Golf solution infers the output side from the largest flattened first-to-last span of a foreground color, then centers each non-background color inside that square. Both references match the generator's intended center-each-color-motif rule.
