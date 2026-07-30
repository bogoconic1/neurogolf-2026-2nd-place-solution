# task020 Semantics

## Sources

- Current champion builder: `solutions_py/task020.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task020.json`
- ARC-GEN task id: `11852cab`
- ARC-DSL task id: `11852cab`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_11852cab.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_11852cab.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task020.py`

## Pattern

The input is a 10x10 grid containing a centered nonzero pattern inside a 5x5 window. The pattern has one center cell and three D4 symmetry orbits around it: diagonal neighbors at radius 1, orthogonal cells at distance 2, and corner cells at radius 2. Exactly one of those orbits may be incomplete in the input: only one cell of the chosen four-cell orbit is kept, while the output restores all four cells of every orbit. The transformation is the D4 closure of the nonzero object: paint the horizontal mirror, vertical mirror, main-diagonal mirror, and anti-diagonal mirror of the object back onto the original grid.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    cells = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != 0]
    rows = [r for r, _, _ in cells]
    cols = [c for _, c, _ in cells]
    cr = (min(rows) + max(rows)) // 2
    cc = (min(cols) + max(cols)) // 2

    for r, c, color in cells:
        dr, dc = r - cr, c - cc
        for rr, cc2 in {
            ( dr,  dc), ( dr, -dc), (-dr,  dc), (-dr, -dc),
            ( dc,  dr), ( dc, -dr), (-dc,  dr), (-dc, -dr),
        }:
            out[cr + rr][cc + cc2] = color
    return out
```

## Generator Constraints

- The generated grid is 10x10.
- The pattern center row and column are independently chosen from 3 through 6, so the whole 5x5 window lies inside rows and columns 1 through 8.
- The central cell, radius-1 diagonal orbit, radius-2 orthogonal orbit, and radius-2 diagonal/corner orbit each have one color chosen from `{1,2,3,4,8}`; colors may repeat across orbits.
- One radius from 1 through 3 is chosen as the incomplete orbit, and one of that orbit's four positions is kept in the input. The output contains all four positions for every orbit.
- Background is 0 and output size is unchanged at 10x10.

## Reference Notes

- ARC-DSL computes all non-background objects, merges them, then paints the horizontal mirror, vertical mirror, main-diagonal mirror, and anti-diagonal mirror of the merged object onto the grid.
- ARC-GEN exposes the concrete 5x5 orbit structure and the center constraints used by compact ONNX solutions.
- The Code Golf 2025 solution repeatedly applies transpose/reversal-style regex substitutions to propagate the one kept cell through the symmetry closure.
