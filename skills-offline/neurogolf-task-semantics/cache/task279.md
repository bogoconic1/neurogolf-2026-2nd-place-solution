# task279 Semantics

## Sources

- Current champion builder: `solutions_py/task279.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task279.json`
- ARC-GEN task id: `b2862040`
- ARC-DSL task id: `b2862040`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b2862040.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b2862040.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task279.py`

## Pattern

The input uses maroon (`9`) as background and blue (`1`) for all visible box/barnacle strokes. Some original boxes were closed cyan (`8`) in the hidden generator state, but the input maps cyan to blue, so the task is to infer which blue components enclose background holes. The output recolors the closed/enclosing blue components to cyan (`8`) and leaves open blue components blue. Background remains maroon.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    # Exterior flood fill through non-blue cells.
    exterior = [[False for _ in range(w)] for _ in range(h)]
    stack = []
    for r in range(h):
        for c in (0, w - 1):
            if grid[r][c] != 1 and not exterior[r][c]:
                exterior[r][c] = True
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r][c] != 1 and not exterior[r][c]:
                exterior[r][c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] != 1 and not exterior[nr][nc]:
                exterior[nr][nc] = True
                stack.append((nr, nc))

    holes = {(r, c) for r in range(h) for c in range(w) if grid[r][c] != 1 and not exterior[r][c]}
    out = [row[:] for row in grid]
    # Recolor blue pixels in components adjacent to enclosed holes.
    seen = [[False for _ in range(w)] for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 1 or seen[r][c]:
                continue
            comp = []
            touches_hole = False
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if (nr, nc) in holes:
                        touches_hole = True
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 1 and not seen[nr][nc]:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if touches_hole:
                for rr, cc in comp:
                    out[rr][cc] = 8
    return out
```

## Generator Constraints

Raw width and height are `10..16`. There are `2..5` non-overlapping rectangular box perimeters, widths and heights `3..5`, separated by at least a two-cell margin. At least one box is open and at least one is closed. Closed boxes use cyan in the hidden output; open boxes use blue and have one perimeter pixel removed. Small optional same-color barnacles may be drawn around box corners/edges and then de-neighborized. The submitted input remaps all cyan to blue, so only colors `1` and `9` appear in the input; output colors are `1`, `8`, and `9`.

## Reference Notes

ARC-DSL filters blue objects that are adjacent to background components not bordering the grid, then fills those blue objects with cyan. Code Golf performs repeated rotations/scans to propagate the closed/open state around boundaries.
