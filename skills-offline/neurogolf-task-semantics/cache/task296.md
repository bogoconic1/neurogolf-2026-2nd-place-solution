# task296 Semantics

## Sources

- Current champion builder: `solutions_py/task296.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task296.json`
- ARC-GEN task id: `bc1d5164`
- ARC-DSL task id: `bc1d5164`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_bc1d5164.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_bc1d5164.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task296.py`

## Pattern

The input is a 5x7 grid embedded in the standard NeuroGolf tensor. All
non-background cells use one shared foreground color. Only the four 2x2 corner
blocks matter:

- top-left rows 0..1, cols 0..1
- top-right rows 0..1, cols 5..6
- bottom-left rows 3..4, cols 0..1
- bottom-right rows 3..4, cols 5..6

The output is a 3x3 grid. Each selected corner block is shifted into the 3x3
canvas: bottom blocks move up by 2 rows, and right blocks move left by 4
columns. The four shifted 2x2 masks are overlaid. Because every foreground
cell has the same color, overlapping colored cells agree; a target cell is the
foreground color if any contributing source cell is colored, otherwise it is
background 0.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(3)] for _ in range(3)]

    for r in range(len(grid)):
        for c in range(len(grid[0])):
            color = grid[r][c]
            if color == 0:
                continue
            if r == 2 or c in (2, 3, 4):
                continue

            rr = r if r < 2 else r - 2
            cc = c if c < 2 else c - 4
            out[rr][cc] = color

    return out
```

## Generator Constraints

ARC-GEN uses fixed input dimensions `height=5`, `width=7` and fixed output
size `3x3`. The random source pixels are chosen only from the four 2x2 corner
blocks; row 2 and columns 2, 3, and 4 are excluded. All chosen pixels receive
one random non-background color. The output starts as all background and fills
the shifted coordinates described above.

Important edge cases:

- output cells can legitimately remain background 0
- multiple source pixels can map to the same output cell
- all mapped foreground pixels have the same color, so overlay order does not
  matter for generated cases
- channel 0/background cannot be discarded in the final output comparison,
  because a valid in-crop target cell may be background

## Reference Notes

ARC-DSL computes `leastcolor(I)` as the shared foreground color, crops four 3x3 windows at `(0,0)`, `(2,0)`, `(4,0)`, and `(2,4)` in DSL coordinate order, then collects the foreground-color positions across those crops and fills a 3x3 canvas. The 3x3 crop formulation is equivalent to overlaying the four valid 2x2 corner source blocks after the generator exclusions.

The Code Golf solution zips the first three columns of the top rows, bottom rows, and right-side suffixes and takes a per-cell maximum. That works because the foreground color is shared and background is 0.
