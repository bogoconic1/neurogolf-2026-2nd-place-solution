# task168 Semantics

## Sources

- Current champion builder: `solutions_py/task168.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task168.json`
- ARC-GEN task id: `6e19193c`
- ARC-DSL task id: `6e19193c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6e19193c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6e19193c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task168.py`

## Pattern

The input is a 10x10 grid with 2 or 3 non-overlapping same-color arrow glyphs on black background. Each glyph starts as a 2x2 colored block with one corner cell black. The missing corner indicates a diagonal direction away from the 2x2 block:

- missing top-left: ray extends up-left
- missing top-right: ray extends up-right
- missing bottom-left: ray extends down-left
- missing bottom-right: ray extends down-right

The output preserves the colored 2x2 body and the black missing corner, and fills the diagonal ray outward from just beyond the missing corner to the grid edge with the glyph color. All arrows have the same nonzero color. The generator prevents ray collisions and keeps glyphs separated.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    color = next(v for row in grid for v in row if v != 0)
    out = [row[:] for row in grid]

    # Find every 2x2 window with exactly three cells of the arrow color and one black corner.
    for r in range(h - 1):
        for c in range(w - 1):
            cells = [(r, c), (r, c + 1), (r + 1, c), (r + 1, c + 1)]
            vals = [grid[rr][cc] for rr, cc in cells]
            if vals.count(color) != 3 or vals.count(0) != 1:
                continue
            hole = cells[vals.index(0)]
            hr, hc = hole
            dr = -1 if hr == r else 1
            dc = -1 if hc == c else 1
            rr, cc = hr + dr, hc + dc
            while 0 <= rr < h and 0 <= cc < w:
                out[rr][cc] = color
                rr += dr
                cc += dc
    return out
```

## Generator Constraints

ARC-GEN uses fixed 10x10 grids. It samples 2 or 3 arrows, all in one random nonzero color. Each arrow anchor row/col is in `1..7`, so the 2x2 body and one-cell-outside direction have room. Arrow directions are unique, glyph bodies are at least four rows/cols apart, and the generated diagonal rays do not collide or overlap in ambiguous ways.

## Reference Notes

The ARC-DSL solver finds the least non-background color, identifies colored objects, detects each object's black corner by looking at neighboring cells that have exactly two object-color neighbors, shoots a ray from the object/corner relationship, fills that ray with the arrow color, then restores the internal black corner cells. The Code Golf solution is a compact recursive/regex/transposition implementation of the same diagonal ray extension.
