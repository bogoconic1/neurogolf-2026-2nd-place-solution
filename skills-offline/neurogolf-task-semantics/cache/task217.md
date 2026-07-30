# task217 Semantics

## Sources

- Current champion builder: `solutions_py/task217.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task217.json`
- ARC-GEN task id: `8f2ea7aa`
- ARC-DSL task id: `8f2ea7aa`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8f2ea7aa.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8f2ea7aa.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task217.py`

## Pattern

The input is a 9x9 grid containing one colored Conway-style sprite embedded by
modulo-3 residue classes. The sprite has logical coordinates inside a 3x3
pattern. One sprite cell is chosen as an anchor, and every sprite cell is drawn
at `(anchor_row * 3 + sprite_row, anchor_col * 3 + sprite_col)` in one nonzero
color. The output is the Kronecker product of the logical 3x3 sprite pattern
with itself: if logical cells `(a,b)` and `(d,e)` are both present, output cell
`(3*a+d, 3*b+e)` is the sprite color. Inactive cells inside the 9x9 output are
background 0; dense NeuroGolf padding outside the 9x9 output is zero-hot.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    color = 0
    present = [[False] * 3 for _ in range(3)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                color = grid[r][c]
                present[r % 3][c % 3] = True

    out = [[0 for _ in range(9)] for _ in range(9)]
    for a in range(3):
        for b in range(3):
            if not present[a][b]:
                continue
            for d in range(3):
                for e in range(3):
                    if present[d][e]:
                        out[3 * a + d][3 * b + e] = color
    return out
```

## Generator Constraints

ARC-GEN uses `size=3`, so input and output ARC grids are both 9x9. It samples a
Conway sprite, selects one of its cells as the anchor, and chooses one nonzero
color. All foreground input pixels share that color. The input placement makes
logical sprite coordinates recoverable from row and column residues modulo 3;
the absolute 3x3 block containing the visible pixels is not necessarily the
source pattern.

## Reference Notes

ARC-DSL extracts the colored object, takes its subgrid, upsamples it by 3, tiles the subgrid 3 by 3, and uses `cellwise` to keep only positions where both the upsampled outer pattern and tiled inner pattern are foreground. The Code Golf solution expresses the same product recursively.
