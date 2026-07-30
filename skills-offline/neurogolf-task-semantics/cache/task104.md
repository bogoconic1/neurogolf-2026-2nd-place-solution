# task104 Semantics

## Sources

- Current champion builder: `solutions_py/task104.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task104.json`
- ARC-GEN task id: `4522001f`
- ARC-DSL task id: `4522001f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4522001f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4522001f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task104.py`

## Pattern

The input is a `3x3` direction marker. The center is red (`2`). Green (`3`)
pixels occupy one of the four 2x2 corner neighborhoods around the center,
including the two adjacent edge cells and the corner cell for that quadrant.
The output is a `9x9` grid containing two `4x4` green blocks on a diagonal. The
orientation/placement of the blocks is determined by the marked quadrant:
quadrant 0/top-left gives blocks at top-left and center, quadrant 1/top-right is
the horizontal mirror, quadrant 2/bottom-left is the vertical mirror, and quadrant
3/bottom-right gives blocks at lower-right and center-shifted lower/right.

## Readable Python Solver

```python
def solve(grid):
    # Quadrant is identified by which corner-neighborhood around center is green.
    if grid[0][0] == 3:
        q = 0
    elif grid[0][2] == 3:
        q = 1
    elif grid[2][0] == 3:
        q = 2
    else:
        q = 3

    row_shift = 1 if q in (2, 3) else 0
    col_shift = 1 if q in (1, 3) else 0
    if q in (0, 3):
        blocks = [(row_shift, col_shift), (row_shift + 4, col_shift + 4)]
    else:
        blocks = [(row_shift, col_shift + 4), (row_shift + 4, col_shift)]

    out = [[0] * 9 for _ in range(9)]
    for br, bc in blocks:
        for dr in range(4):
            for dc in range(4):
                out[br + dr][bc + dc] = 3
    return out
```

## Generator Constraints

ARC-GEN uses exactly four possibilities (`quadrant` in `0..3`) and fixed input
size `3`. The output size is always `9x9`. The green input cells are deterministic
for the quadrant: top-left, top-right, bottom-left, or bottom-right corner
neighborhood around the red center. Output consists only of black (`0`) and green
(`3`) and is formed by rotating/mirroring a fixed pair of `4x4` blocks.

## Reference Notes

ARC-DSL identifies which of the three corner coordinates `(0,2)`, `(2,2)`, and `(2,0)` is contained in the green object, paints a base 9x9 two-block shape, and selects the appropriate rotation. Code Golf 2025 encodes the same four outputs as row constants with row/column reversal controlled by specific input cells.
