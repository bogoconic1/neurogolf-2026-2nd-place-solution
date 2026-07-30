# task365 Semantics

## Sources

- Current champion builder: `solutions_py/task365.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task365.json`
- ARC-GEN task id: `e50d258f`
- ARC-DSL task id: `e50d258f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e50d258f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e50d258f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task365.py`

## Pattern

The input is a 10x10 grid with black background (`0`) and two or three non-overlapping colored rectangles. Each rectangle has width and height from 3 through 6. Its cells are mostly blue/cyan-like colors (`1` or `8`) with a small number of red cells (`2`) sprinkled inside. The red counts are distinct and assigned in descending order by generated box index, so exactly one rectangle has the most red pixels. The output is the tight subgrid of the rectangle with the largest number of red cells, preserving all colors inside that rectangle.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] == 0:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] != 0:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            comps.append(cells)

    best = max(comps, key=lambda cells: sum(1 for r, c in cells if grid[r][c] == 2))
    r0 = min(r for r, _ in best)
    r1 = max(r for r, _ in best)
    c0 = min(c for _, c in best)
    c1 = max(c for _, c in best)
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]
```

## Generator Constraints

- Input size is always 10x10.
- There are two or three rectangles: `num_boxes = min(3, randint(2, 5))`.
- Rectangle widths and heights are each 3 through 6.
- Rectangles are non-overlapping with a one-cell separation margin according to `common.overlaps(..., 1)`.
- Non-red cells inside rectangles are randomly color `1` or `8`.
- Red cells use color `2`; each rectangle receives a distinct red count sampled from `1..4`, sorted descending by rectangle index.
- The expected output is exactly the first/generated rectangle, equivalently the unique rectangle with maximum red count.

## Reference Notes

The ARC-DSL solver vertically concatenates extra black rows, extracts objects, and selects the object maximizing `colorcount(TWO)`, then returns `subgrid` for that object. The padding is a DSL convenience to make object extraction robust at the lower boundary. The Code Golf solution brute-force scans candidate windows and maximizes red count with tie-breakers that prefer nonzero rectangular crops; under the generator the max-red rectangle is unique.
