# task346 Semantics

## Sources

- Current champion builder: `solutions_py/task346.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task346.json`
- ARC-GEN task id: `d9fac9be`
- ARC-DSL task id: `d9fac9be`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d9fac9be.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d9fac9be.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task346.py`

## Pattern

The input is a rectangular grid, padded to the standard NeuroGolf tensor, with background `0` and exactly two non-background colors. One non-background color forms an almost solid `3x3` block: all eight cells around the center have the same color, while the center cell has the other non-background color. The output is a `1x1` grid containing the center color, i.e. the non-background color that is not the color of the surrounding `3x3` ring / largest object.

Random noise of both non-background colors may appear elsewhere, so pure global frequency is not by itself the semantic rule. The key local witness is the `3x3` window with eight cells of one color and the center cell of the other.

## Readable Python Solver

```python
def solve(grid):
    colors = sorted({v for row in grid for v in row if v != 0})
    h, w = len(grid), len(grid[0])

    # Prefer the explicit 3x3 ring witness from the generator.
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = grid[r][c]
            if center == 0:
                continue
            ring = [
                grid[r + dr][c + dc]
                for dr in (-1, 0, 1)
                for dc in (-1, 0, 1)
                if dr or dc
            ]
            if len(set(ring)) == 1 and ring[0] != 0 and ring[0] != center:
                return [[center]]

    # Equivalent ARC-DSL fallback: find the largest nonzero same-color object
    # and return the other nonzero color.
    seen = set()
    best_color, best_size = None, -1
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            seen.add((r, c))
            size = 0
            while stack:
                rr, cc = stack.pop()
                size += 1
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and (nr, nc) not in seen
                        and grid[nr][nc] == color
                    ):
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            if size > best_size:
                best_color, best_size = color, size

    return [[next(color for color in colors if color != best_color)]]
```

## Generator Constraints

- Width and height are independently sampled from `5..12`.
- Exactly two non-background colors are sampled. The initial bitmap receives sparse random pixels, about 20% density, colored from those two colors.
- A center coordinate is sampled with a two-cell margin: `row in 2..height-3`, `col in 2..width-3`.
- The full `3x3` neighborhood around that center is first filled with `color_list[1]`; then the center cell is overwritten with `color_list[0]`.
- The output is `[[center]]`, where `center` is `color_list[0]`.
- Because the center has a two-cell margin and the grid is at most `12x12`, the decisive `3x3` witness always lies inside rows/cols `1..10` of the padded `30x30` NeuroGolf tensor.
- Noise can change global color counts, including cases where the center color is globally more frequent than the ring color. A solver must detect the ring/local object, not only choose a fixed count order.

## Reference Notes

- ARC-DSL computes the palette, extracts objects, takes the largest object, gets its color, removes background from the palette, and returns the other non-background color as a `1x1` canvas.
- ARC-GEN makes the largest-object interpretation concrete by constructing an eight-cell ring/object in one color around a center cell of the other color.
- The Code Golf 2025 one-liner chooses a low-frequency color after a fixed flattening/slicing trick; it is a compact exploit for the generated examples, but the clearer semantic rule is still "return the center color inside the 3x3 ring" / "return the color other than the largest object color."
- If the major color has the ring, output the other color; otherwise output the major color.
