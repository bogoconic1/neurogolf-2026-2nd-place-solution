# task314 Semantics

## Sources

- Current champion builder: `solutions_py/task314.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task314.json`
- ARC-GEN task id: `cbded52d`
- ARC-DSL task id: `cbded52d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_cbded52d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_cbded52d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task314.py`

## Pattern

The input is an 8x8 grid with blue background and black separator rows/columns wherever `row % 3 == 2` or `col % 3 == 2`. The non-separator area represents a 3x3 lattice of 2x2 cells. Colored singleton pixels appear at one of four offsets inside a 2x2 cell. For each pair of same-offset/same-color singleton objects that share a logical row or logical column with exactly one missing cell between them, fill the corresponding middle singleton position with that same color. Existing colored pixels and black separators are preserved.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    # logical positions r,c in 0..2 and offsets dr,dc in {0,1}
    seen = {}
    for r in range(3):
        for c in range(3):
            for dr in range(2):
                for dc in range(2):
                    y, x = 3 * r + dr, 3 * c + dc
                    color = grid[y][x]
                    if color not in (0, 1):  # non-background, non-blue/black separator color
                        seen.setdefault((dr, dc, color), []).append((r, c))
    for (dr, dc, color), pts in seen.items():
        pts = set(pts)
        for r in range(3):
            if (r, 0) in pts and (r, 2) in pts:
                out[3 * r + dr][3 * 1 + dc] = color
        for c in range(3):
            if (0, c) in pts and (2, c) in pts:
                out[3 * 1 + dr][3 * c + dc] = color
    return out
```

## Generator Constraints

ARC-GEN fixes the grid to 8x8. Separator cells (`r % 3 == 2` or `c % 3 == 2`) are black in both input and output; other background is blue. It chooses either offset pair `[0,3]` or `[1,2]`, then may place same-offset colored singleton endpoints at corners, horizontal endpoints, or vertical endpoints. Colors are four random colors excluding blue; the generated examples only require completing missing middle singleton positions between matching same-offset endpoints in the 3x3 logical lattice.

## Reference Notes

The ARC-DSL solver finds singleton objects, pairs them, filters to pairs that share a row or column, connects their centers, and paints those connection points with the pair color. Because all objects are singleton offset positions inside the fixed lattice, the connection adds only the missing midpoint singleton. The Code Golf solution uses row/column recurrences over the 8x8 layout to fill the center when matching endpoints appear three cells apart.
