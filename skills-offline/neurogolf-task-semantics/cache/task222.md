# task222 Semantics

## Sources

- Current champion builder: `solutions_py/task222.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task222.json`
- ARC-GEN task id: `91714a58`
- ARC-DSL task id: `91714a58`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_91714a58.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_91714a58.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task222.py`

## Pattern

The input is a 16x16 noisy color grid containing one solid axis-aligned rectangle of a single color. The rectangle has width and height in `2..8`, area between 9 and 16 cells, and is placed away from the grid boundary. Random noise may contain the same colors, but the generator rejects cases where same-color neighbors outside the rectangle extend any entire side. The output keeps only the target rectangle and turns every other cell black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    best = None
    for r0 in range(h):
        for c0 in range(w):
            color = grid[r0][c0]
            if color == 0:
                continue
            for r1 in range(r0 + 1, h + 1):
                for c1 in range(c0 + 1, w + 1):
                    area = (r1 - r0) * (c1 - c0)
                    if area < 9 or area > 16:
                        continue
                    ok = all(grid[r][c] == color for r in range(r0, r1) for c in range(c0, c1))
                    if ok and (best is None or area > best[0]):
                        best = (area, color, r0, c0, r1, c1)
    out = [[0 for _ in row] for row in grid]
    if best:
        _, color, r0, c0, r1, c1 = best
        for r in range(r0, r1):
            for c in range(c0, c1):
                out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN uses fixed `size=16`. The hidden rectangle color is a random color and may be black only if `common.random_color()` allows it, and the task logic treats labels 0..9 generally. Width and height are each `2..8`, with area constrained to `9..16`. The rectangle top-left row/col are at least one cell from the grid edge. Noise is random at about 50% density, with random colors, then the rectangle overwrites its cells. The generator rejects accidental same-color neighbor runs that would extend the rectangle along a full side.

## Reference Notes

The ARC-DSL solver finds connected color objects, takes the largest by size, gets its majority/most color, paints that object on a black canvas, then clears cells whose same-color neighbor count is greater than three. For a filled rectangle, that preserves the rectangle while removing noisy same-color overgrowth. The Code Golf solution repeatedly transposes/filters and keeps colors that appear as a contiguous rectangle with enough count, matching the same target-object extraction.
