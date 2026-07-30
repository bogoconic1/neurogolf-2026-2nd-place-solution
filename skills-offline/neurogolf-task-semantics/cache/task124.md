# task124 Semantics

## Sources

- Current champion builder: `solutions_py/task124.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task124.json`
- ARC-GEN task id: `53b68214`
- ARC-DSL task id: `53b68214`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_53b68214.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_53b68214.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task124.py`

## Pattern

The input is a partial top crop of a periodic single-color pattern on a `10`-column grid. Input height is `5..8`; output is always `10 x 10`. The nonzero cells all share one color. A small sprite with bounding height `tall in 1..3` and width `wide in 1..3` is repeated downward every `tall` rows. In the normal case each repetition keeps the same columns. In the diagonal case, allowed only for height-2 sprites in the generator, each repetition also shifts right by `wide - 1` columns. Cells outside the `10 x 10` canvas are clipped. The task is to continue the observed vertical or diagonal periodic sequence from the input crop down to row 9.

## Readable Python Solver

```python
def solve(grid):
    H, W = len(grid), len(grid[0])
    color = next(v for row in grid for v in row if v)
    seen = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v}

    best = None
    for tall in range(1, 4):
        for shift in range(0, 3):
            base = set()
            ok = True
            for r, c in seen:
                q = r // tall
                br = r % tall
                bc = c - q * shift
                if bc < 0:
                    ok = False
                    break
                base.add((br, bc))
            if not ok or not base:
                continue

            recon = set()
            for q in range(10):
                for br, bc in base:
                    r = q * tall + br
                    c = q * shift + bc
                    if r < H and 0 <= c < W:
                        recon.add((r, c))
            if recon == seen:
                # Prefer the shortest vertical period, then no diagonal shift,
                # matching the generator/DSL period choice on ambiguous columns.
                key = (tall, shift != 0, shift, len(base))
                if best is None or key < best[0]:
                    best = (key, tall, shift, base)

    _, tall, shift, base = best
    out = [[0 for _ in range(10)] for _ in range(10)]
    for q in range(10):
        for br, bc in base:
            r = q * tall + br
            c = q * shift + bc
            if 0 <= r < 10 and 0 <= c < 10:
                out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN fixes width to `10` and samples input height `5..8`. The sprite is made by `common.conway_sprite(wide, tall)` with `wide,tall in {2,3}` except smaller heights force `tall=2` and height `<6` also forces `wide=1`; hand-authored validation examples include a one-cell/tall-1 vertical column. The generator requires the sprite to occupy every row of its bounding height and every column of its bounding width. The vertical offset/left margin is `0..3`; color is any nonzero ARC color. Diagonal repetition is sampled only when `tall <= 2`; otherwise the horizontal shift per period is zero. The output draws the same repeated pattern into a full `10 x 10` canvas and clips cells that go beyond the right or bottom boundary.

## Reference Notes

The ARC-DSL solver extracts the single object, computes its vertical period with `vperiod`, and either repeats it downward by multiples of that period or, for non-portrait/diagonal cases, shifts it once by `(height, width - 1)` before painting into a square canvas. The compact Code Golf solution recursively tries rotations/period parameters until the 10-row continuation is stable. The references agree that the core state is the observed nonzero object, its vertical period, and whether its next copy shifts horizontally.
