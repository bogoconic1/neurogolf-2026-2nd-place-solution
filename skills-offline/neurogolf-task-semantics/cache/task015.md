# task015 Semantics

## Sources

- Current champion builder: `solutions_py/task015.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task015.json`
- ARC-GEN task id: `0ca9ddb6`
- ARC-DSL task id: `0ca9ddb6`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_0ca9ddb6.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_0ca9ddb6.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task015.py`

## Pattern

The input is a 9x9 star field embedded in the standard padded NeuroGolf one-hot tensor. Nonzero input cells are isolated stars in colors 1, 2, 6, or 8. Colors 6 and 8 remain unchanged. Color 1 stars add color-7 twinkles on their four orthogonal neighbors. Color 2 stars add color-4 twinkles on their four diagonal neighbors. Original star cells keep their original colors, background stays color 0, and the padded area outside the 9x9 board stays zero-hot.

The generator moves color 1 and 2 stars away from the 9x9 border before emitting twinkles, and rejects stars whose own footprint would overlap an earlier star footprint. That means twinkle writes do not need conflict resolution in accepted examples.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    h, w = len(grid), len(grid[0])
    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 1:
                for dr, dc in ((1, 0), (0, 1), (-1, 0), (0, -1)):
                    out[r + dr][c + dc] = 7
            elif color == 2:
                for dr, dc in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                    out[r + dr][c + dc] = 4
    return out
```

## Generator Constraints

- Grid size is fixed at 9x9.
- The random generator samples 1..6 star positions from the 9x9 board.
- Star colors are independently chosen from `{1, 2, 6, 8}`.
- Color 1 and color 2 stars are shifted away from the outer border so all twinkle neighbors remain in-bounds.
- A color 1 or color 2 star has radius 1 for overlap checks; colors 6 and 8 have radius 0. A candidate star is kept only if its footprint does not overlap the footprint of any earlier kept star in row and column distance.
- Color 1 emits rook-neighbor color 7; color 2 emits diagonal-neighbor color 4.
- Color 6 and color 8 only copy themselves.

## Reference Notes

- ARC-DSL expresses the rule directly: get all color-1 cells, get all color-2 cells, fill direct neighbors of color 1 with 7, then fill diagonal neighbors of color 2 with 4.
- ARC-GEN confirms that twinklers are colors 1 and 2, with twinkle colors 7 and 4 respectively, and that generated twinkles stay inside the board.
- The Code Golf solution is a terse recursive transpose/neighbor fill, agreeing with the same color-1 rook and color-2 diagonal twinkle semantics.
