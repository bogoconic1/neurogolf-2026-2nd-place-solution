# task196 Semantics

## Sources

- Current champion builder: `solutions_py/task196.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task196.json`
- ARC-GEN task id: `810b9b61`
- ARC-DSL task id: `810b9b61`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_810b9b61.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_810b9b61.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task196.py`

## Pattern

The input is a square grid containing several disjoint blue (`1`) axis-aligned rectangle perimeters, line segments, singleton/flat shapes, and some rectangle perimeters with one missing border cell. The output keeps the same grid size. Every completely closed hollow rectangle with width at least 3 and height at least 3 is recolored from blue (`1`) to green (`3`). Background stays black (`0`). Broken rectangles, rectangles with a removed perimeter cell, flat/1-thick shapes, and line/singleton shapes remain blue.

Equivalently: for each blue connected component, ignore components that are pure horizontal/vertical lines. For every remaining component, compare the blue cell set with its bounding-box border. If the component exactly equals the full border of its bounding box, recolor all its cells green; otherwise leave it blue.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    seen = [[False] * w for _ in range(h)]

    for sr in range(h):
        for sc in range(w):
            if seen[sr][sc] or grid[sr][sc] != 1:
                continue
            stack = [(sr, sc)]
            seen[sr][sc] = True
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 1:
                        seen[nr][nc] = True
                        stack.append((nr, nc))

            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            if r1 == r0 or c1 == c0:
                continue

            border = set()
            for r in range(r0, r1 + 1):
                border.add((r, c0))
                border.add((r, c1))
            for c in range(c0, c1 + 1):
                border.add((r0, c))
                border.add((r1, c))

            if set(comp) == border:
                for r, c in comp:
                    out[r][c] = 3
    return out
```

## Generator Constraints

- Grid size is `3 * randint(3,5)`, so only `9x9`, `12x12`, or `15x15` inputs occur in ARC-GEN. NeuroGolf still uses the standard `[1,10,30,30]` canvas.
- Number of boxes is between `size//4` and `size//2`.
- Rectangles/objects are non-overlapping with at least one-cell spacing (`common.overlaps(..., 1)` rejects close overlaps).
- Widths and heights are each sampled from `1..5`.
- Width-1 or height-1 objects are never gapped and remain blue.
- Some width/height >=3 boxes are deliberately kept closed; the generator requires at least one such closed box.
- Gaps are single removed border cells. For boxes with both dimensions at least 3, gaps are never corners: they are on a side but not at a corner. For 2xH or Wx2 boxes, gaps can be corners.
- Only blue input objects and black background are used; output introduces green only for closed valid rectangles.

## Reference Notes

The ARC-DSL solver extracts blue objects, removes pure vertical/horizontal line objects, then keeps only non-line objects whose cell set equals their bounding box border. Those selected cells are filled green.

The Code Golf 2025 solution is a compact iterative transformation, but it implements the same idea: distinguish closed box perimeters from broken/flat shapes and recolor only closed boxes.
