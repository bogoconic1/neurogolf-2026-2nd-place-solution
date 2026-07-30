# task137 Semantics

## Sources

- Current champion builder: `solutions_py/task137.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task137.json`
- ARC-GEN task id: `5c2c9af4`
- ARC-DSL task id: `5c2c9af4`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_5c2c9af4.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_5c2c9af4.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task137.py`

## Pattern

The input is a square grid of size 20..30 padded into the standard 30x30 NeuroGolf tensor. It contains exactly three non-black pixels of one color: a center pixel and two opposite diagonal pixels at Chebyshev distance `spacing` from the center. The two outer pixels may lie on either diagonal.

The output keeps those pixels and fills every square outline centered at the center whose radius is a positive multiple of `spacing`, continuing outward until the square outline is completely outside the grid. Equivalently, for every visible cell `(r, c)`, color it if `max(abs(r-center_r), abs(c-center_c)) % spacing == 0`; otherwise leave it black. The color is the single nonzero input color. Padded cells outside the original square grid stay zero-hot/empty.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    pts = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != 0]
    color = pts[0][2]
    rows = sorted(r for r, c, v in pts)
    cols = sorted(c for r, c, v in pts)
    center_r = rows[1]
    center_c = cols[1]
    spacing = (rows[-1] - rows[0]) // 2

    out = [[0 for _ in row] for row in grid]
    s = spacing
    while not (center_r - s < 0 and center_r + s >= h and
               center_c - s < 0 and center_c + s >= w):
        for i in range(-s, s + 1):
            for r, c in ((center_r - s, center_c + i),
                         (center_r + s, center_c + i),
                         (center_r + i, center_c - s),
                         (center_r + i, center_c + s)):
                if 0 <= r < h and 0 <= c < w:
                    out[r][c] = color
        s += spacing
    out[center_r][center_c] = color
    return out
```

## Generator Constraints

The grid size is random from 20..30. `spacing` is random from 2 through `size // 4`. The center row and column are chosen so both outer diagonal pixels are in bounds: `spacing <= row <= size-spacing-1` and the same for `col`. The `flip` bit chooses whether the two outer pixels lie on the `\` diagonal or the `/` diagonal. The color is any nonzero ARC color.

There are exactly three colored input pixels and all have the same color. The original grid area is square; the standard NeuroGolf tensor has padded zero-hot cells outside that square.

## Reference Notes

ARC-DSL finds the least/nonzero color, takes the object center and upper-left corner, uses their difference as the spacing vector, creates box outlines at multiples of that vector, shifts them to the center, and fills them with the object color.

The Code Golf 2025 solution also computes the middle colored pixel, then colors cells according to a modulo of the Chebyshev distance from the center. It confirms that orientation of the two outer pixels does not matter for output after spacing and center are known.
