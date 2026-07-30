# task110 Semantics

## Sources

- Current champion builder: `solutions_py/task110.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task110.json`
- ARC-GEN task id: `484b58aa`
- ARC-DSL task id: `484b58aa`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_484b58aa.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_484b58aa.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task110.py`

## Pattern

The logical input and output are fixed `29x29` grids. The hidden complete grid is
a periodic color pattern. The input is the same pattern with five black
rectangular cutouts; the output restores the original colors inside those black
rectangles.

The generator builds the pattern from a modular radial formula or from an
explicit color table. The period length is between `4` and `9`, with an offset,
and colors are nonzero. Black is used only for the cutouts in the input, not in
the target pattern.

## Readable Python Solver

```python
def solve(grid):
    # Reference-style algorithm: infer row/column periodicity from visible
    # nonblack cells and fill each black cutout cell from any compatible source
    # row/column position in the same hidden period class.
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    def compatible_rows(r1, r2):
        for c in range(w):
            a, b = grid[r1][c], grid[r2][c]
            if a and b and a != b:
                return False
        return True

    for r in range(h):
        for c in range(w):
            if out[r][c] != 0:
                continue
            for rr in range(h):
                if not compatible_rows(r, rr):
                    continue
                if grid[rr][c] != 0:
                    out[r][c] = grid[rr][c]
                    break
    return out
```

## Generator Constraints

- Grid size is fixed at `29x29`.
- `length` is `4..mod`, and the random branch has `mod` in `4..9`; public
  examples include larger explicit `mod` values with provided color tables.
- Five black rectangles are placed; each width and height is `2..5` in the
  random branch.
- The target output has no black cutouts; every cell is a nonzero pattern color.
- The visible rows/columns contain enough periodic evidence to recover the
  missing cells by compatibility.

## Reference Notes

ARC-DSL removes the black cutouts, finds row and column periods from near-edge crops, shifts the visible nonblack object by period multiples, and paints those shifted copies back over the input.
