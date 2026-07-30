# task398 Semantics

## Sources

- Current champion builder: `solutions_py/task398.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task398.json`
- ARC-GEN task id: `feca6190`
- ARC-DSL task id: `feca6190`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_feca6190.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_feca6190.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task398.py`

## Pattern

The input is a single row of length 5 containing a palette: zero entries are ignored and nonzero entries are colors to paint. Let `n` be the number of nonzero palette entries. The output is a square of size `5*n` by `5*n`. For each palette position `p` in `0..4` with nonzero color `color`, paint a descending anti-diagonal segment of that color. In generator coordinates, cells satisfy `r + c = (5*n - 1) + p` with `r >= p`; equivalently the nonzero palette row is projected into parallel anti-diagonals on the square. All other output cells are black.

The NeuroGolf graph must still emit the dense `[1,10,30,30]` one-hot tensor; the ARC output square is top-left aligned and padded outside.

## Readable Python Solver

```python
def solve(grid):
    palette = grid[0]
    n = sum(1 for color in palette if color != 0)
    size = 5 * n
    out = [[0 for _ in range(size)] for _ in range(size)]
    for p, color in enumerate(palette):
        if color == 0:
            continue
        for r in range(p, size):
            c = size - 1 + p - r
            if 0 <= c < size:
                out[r][c] = color
    return out
```

## Generator Constraints

- Input is always a `1x5` row.
- The number of nonzero colors is randomly chosen from 1 through 5.
- `colors` is padded with zeros to length 5 and shuffled, so nonzero palette positions may appear anywhere in the row.
- Output square size is exactly `5 * num_nonzero_colors`, so possible sizes are 5, 10, 15, 20, and 25.
- Nonzero colors are ordinary ARC colors in `1..9`; zeros mean no diagonal for that palette position.
- Each nonzero palette entry paints one anti-diagonal segment. Later entries can overwrite earlier cells only if diagonals intersect, but these anti-diagonal sums are distinct by palette position, so there is no conflict between different `p` values.

## Reference Notes

- ARC-DSL counts the nonzero input objects, creates a square canvas of side `5*n`, shoots a diagonal ray from each object center, recolors the ray by the object's color, paints all rays, and horizontally mirrors the result. This matches the generator's anti-diagonal output.
- ARC-GEN gives the exact row length, output size rule, color constraints, and diagonal formula.
- Code Golf 2025 builds a padded row `m` of length `5*n`, repeatedly shifts it right by one with leading zero, stacks the rows, and reverses vertically, which is another way to draw the same anti-diagonal palette map.
