# task306 Semantics

## Sources

- Current champion builder: `solutions_py/task306.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task306.json`
- ARC-GEN task id: `c444b776`
- ARC-DSL task id: `c444b776`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c444b776.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c444b776.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task306.py`

## Pattern

The grid is made from 9x9 tiles separated by yellow (`4`) grid lines. There are always two tile rows and one to three tile columns, giving logical sizes `19x9`, `19x19`, or `19x29`. Exactly one tile contains a sparse colored pattern using 3 or 4 non-yellow colors. All other tile interiors are initially black. The output copies the colored pattern into every tile interior while preserving the yellow separator row/columns and leaving padded space outside the logical grid empty.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    # Recover the non-empty 9x9 pattern by overlaying same residues from all tiles.
    pattern = [[0 for _ in range(9)] for _ in range(9)]
    for r in range(h):
        if r % 10 == 9:
            continue
        for c in range(w):
            if c % 10 == 9:
                continue
            v = grid[r][c]
            if v != 0:
                pattern[r % 10][c % 10] = v

    for base_r in range(0, h, 10):
        for base_c in range(0, w, 10):
            for r in range(9):
                for c in range(9):
                    out[base_r + r][base_c + c] = pattern[r][c]
    return out
```

## Generator Constraints

- Tile size is fixed at `9`; separators are every tenth row/column and are yellow (`4`).
- Height is fixed at two tile rows, so the logical height is always `19`.
- Width is one to three tile columns, so the logical width is `9`, `19`, or `29`.
- A random nonempty sparse set of pixels is sampled in a 9x9 pattern with density around `0.125`.
- Pattern colors are sampled from 3 or 4 random non-yellow colors. The same pattern and colors are copied to every output tile.
- Exactly one tile, selected by `quadrant`, contains the pattern in the input; all output tiles contain it.

## Reference Notes

The ARC-DSL solver finds black objects, selects the smallest black tile interior, extracts the corresponding 9x9 pattern object from the input, normalizes it, and shifts that pattern to every black tile backdrop before painting. The Code Golf solution repeatedly overlays rows shifted by 10 and transposes/reverses to propagate the pattern across the grid. Both confirm that the rule is residue-class pattern copying across separator-delimited tiles.
