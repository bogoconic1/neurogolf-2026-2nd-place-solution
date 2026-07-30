# task139 Semantics

## Sources

- Current champion builder: `solutions_py/task139.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task139.json`
- ARC-GEN task id: `60b61512`
- ARC-DSL task id: `60b61512`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_60b61512.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_60b61512.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task139.py`

## Pattern

The input is a fixed 9x9 grid containing two yellow diagonal-connected mini-sprites inside two fixed 3x3 boxes. In canonical orientation the first box is at rows `1..3`, columns `0..2`, and the second box is at rows `4..6`, columns `5..7`. The generator may apply an inverted transpose to both input and output, which swaps the apparent layout.

The output keeps every original yellow pixel and fills the missing cells inside each sprite's 3x3 box with orange (`7`). Cells outside the two 3x3 boxes remain black. In the standard NeuroGolf output this is padded to `[1,10,30,30]`.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)

    def transpose_inverted(g):
        # ARC-GEN/common transpose_inverted is equivalent to reversing both axes
        # after transpose for this task's fixed examples. This helper is only
        # a readable sketch of the orientation branch, not submitted code.
        return [list(row) for row in zip(*g[::-1])][::-1]

    # Detect orientation by whether the lower/right 3x3 box appears in the
    # canonical bottom-right region or in its inverted-transpose counterpart.
    # A robust implementation can simply evaluate both orientations and choose
    # the one whose fixed boxes cover all yellow pixels.
    candidates = []
    for xpose in (False, True):
        work = transpose_inverted(grid) if xpose else [row[:] for row in grid]
        out = [[0 for _ in range(n)] for _ in range(n)]
        for r0, c0 in ((1, 0), (4, 5)):
            for dr in range(3):
                for dc in range(3):
                    r, c = r0 + dr, c0 + dc
                    out[r][c] = 4 if work[r][c] == 4 else 7
        # A valid orientation covers all yellow pixels from the input.
        ok = all((work[r][c] == 0) or (out[r][c] == 4)
                 for r in range(n) for c in range(n))
        candidates.append((ok, out, xpose))

    _, out, xpose = max(candidates, key=lambda t: t[0])
    return transpose_inverted(out) if xpose else out
```

## Generator Constraints

The grid size is fixed at 9x9 and the mini-sprite size is fixed at 3x3. The two sprite masks are generated independently with `common.conway_sprite()` until each mask is diagonally connected. Both masks are drawn in yellow. Canonical sprite 0 is shifted by `(row+1, col+0)` and canonical sprite 1 by `(row+4, col+5)`. After drawing yellow pixels, the generator visits every cell of both 3x3 boxes and turns empty cells orange in the output only. The optional `xpose` bit applies `common.transpose_inverted` to both input and output.

There are no other colors in the input. The output only uses black, yellow, and orange.

## Reference Notes

The ARC-DSL solver extracts the yellow objects, takes each object's `delta` footprint, and fills those missing footprint cells with orange (`SEVEN`). This matches the box-completion rule: keep the yellow sprite pixels, fill each object's 3x3 bounding box complement with orange.

The Code Golf 2025 solution is a recursive transpose/inverted-orientation trick using a bit mask constant. It agrees that the task is orientation-sensitive but fixed-layout: infer the orientation, then fill the missing cells of the two 3x3 sprite boxes with orange.
