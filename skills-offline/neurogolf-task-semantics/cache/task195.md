# task195 Semantics

## Sources

- Current champion builder: `solutions_py/task195.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task195.json`
- ARC-GEN task id: `80af3007`
- ARC-DSL task id: `80af3007`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_80af3007.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_80af3007.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task195.py`

## Pattern

The input contains a single gray object: a 3x3 binary Conway sprite scaled up by a factor of 3, so every active sprite cell is a solid 3x3 gray block. The scaled object is placed at a random offset inside a black grid.

Recover the underlying 3x3 binary sprite mask `S`. The output is a 9x9 gray pattern where each active macro-cell `(r, c)` of `S` contains a copy of `S`; inactive macro-cells are black. Equivalently:

```text
out[r*3 + rr][c*3 + cc] = gray iff S[r][c] and S[rr][cc]
```

All cells outside the 9x9 output are zero-hot in the NeuroGolf canvas.

## Readable Python Solver

```python
def solve(grid):
    pts = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 5]
    r0, c0 = min(r for r, _ in pts), min(c for _, c in pts)
    mask = [[0] * 3 for _ in range(3)]
    for r, c in pts:
        mask[(r - r0) // 3][(c - c0) // 3] = 1

    out = [[0] * 9 for _ in range(9)]
    for br in range(3):
        for bc in range(3):
            if not mask[br][bc]:
                continue
            for r in range(3):
                for c in range(3):
                    if mask[r][c]:
                        out[br * 3 + r][bc * 3 + c] = 5
    return out
```

## Generator Constraints

- Input width/height are `18+offset` and `16+offset` where `offset in {-1,0,1}`.
- The sprite mask comes from `common.conway_sprite()`, a 3x3 binary pattern with several active cells.
- The scaled input object is placed with `rowoffset` and `coloffset` at least 1 and with enough room for the 9x9 scaled object.
- Only color gray (`5`) is used.
- The output is always a 9x9 grid.

## Reference Notes

- ARC-DSL extracts the object subgrid, upscales the subgrid by 3, tiles the original subgrid into a 3x3 block layout, intersects them cellwise, then downscales by 3. This is equivalent to the self-Kronecker/outer-product mask.
- The Code Golf 2025 solution trims empty rows/columns, samples every third row/column to recover the 3x3 mask, then uses nested comprehensions with bitwise `&` to build the 9x9 self-product.
