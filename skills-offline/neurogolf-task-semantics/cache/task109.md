# task109 Semantics

## Sources

- Current champion builder: `solutions_py/task109.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task109.json`
- ARC-GEN task id: `47c1f68c`
- ARC-DSL task id: `47c1f68c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_47c1f68c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_47c1f68c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task109.py`

## Pattern

The logical ARC input is an odd square of size `2*k + 1`, with `k` in
`{3, 4, 5, 6}`. A full separator cross occupies row `k` and column `k` in a
foreground line color. A colored sprite lives in the upper-left `k x k`
quadrant, using one color different from the line color.

The logical output is a `2*k x 2*k` square. For each sprite cell `(r, c)` in
the upper-left quadrant, write the line color at all four mirror positions:
`(r, c)`, `(r, 2*k-c-1)`, `(2*k-r-1, c)`, and
`(2*k-r-1, 2*k-c-1)`. All other cells are black. The NeuroGolf graph still
emits a dense `[1, 10, 30, 30]` tensor; cells outside the active `2*k x 2*k`
output area must remain nonpositive in all channels.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    k = n // 2
    line_color = grid[k][k]
    out = [[0 for _ in range(2 * k)] for _ in range(2 * k)]
    for r in range(k):
        for c in range(k):
            if grid[r][c] != 0:
                out[r][c] = line_color
                out[r][2 * k - c - 1] = line_color
                out[2 * k - r - 1][c] = line_color
                out[2 * k - r - 1][2 * k - c - 1] = line_color
    return out
```

## Generator Constraints

- `k` is `3..6`; input sizes are `7x7`, `9x9`, `11x11`, or `13x13`.
- The sprite is generated inside the upper-left quadrant with width and height
  at least `3`, each at most `k`, and can be nudged down by one when height is
  smaller than `k`.
- The sprite uses one nonzero color; the separator cross uses another nonzero
  color.
- The separator cross is complete across the middle row and middle column.
- The output size is exactly `2*k x 2*k` before NeuroGolf zero padding.

## Reference Notes

ARC-DSL identifies the background/cross, mirrors the input vertically and horizontally, compresses away the separator cross, and replaces sprite color with the line color. This agrees with the generator's four-way mirror rule.

The Code Golf 2025 solution recursively builds the top half plus its mirror and uses the center-column color as the emitted foreground color. It is consistent with the fixed `k` candidates.
