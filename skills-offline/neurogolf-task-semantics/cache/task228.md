# task228 Semantics

## Sources

- Current champion builder: `solutions_py/task228.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task228.json`
- ARC-GEN task id: `952a094c`
- ARC-DSL task id: `952a094c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_952a094c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_952a094c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task228.py`

## Pattern

The input is a 10x10 active grid embedded in the standard 30x30 one-hot tensor. It contains one hollow rectangular frame in a non-black color and four single-cell colored markers at the rectangle interior corners. The output removes those four marker cells, leaving the hollow frame and black interior, then paints the four outside corners of the frame outbox with the opposite marker colors: top-left outside gets the bottom-right marker, top-right gets bottom-left, bottom-left gets top-right, and bottom-right gets top-left.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    # Find the frame color as the non-black color with the largest count.
    counts = {}
    for row in grid:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0) + 1
    frame = max(counts, key=counts.get)

    frame_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == frame]
    top = min(r for r, c in frame_cells)
    bottom = max(r for r, c in frame_cells)
    left = min(c for r, c in frame_cells)
    right = max(c for r, c in frame_cells)

    out = [row[:] for row in grid]
    tl = grid[top + 1][left + 1]
    tr = grid[top + 1][right - 1]
    bl = grid[bottom - 1][left + 1]
    br = grid[bottom - 1][right - 1]

    for r, c in [(top + 1, left + 1), (top + 1, right - 1),
                 (bottom - 1, left + 1), (bottom - 1, right - 1)]:
        out[r][c] = 0

    out[top - 1][left - 1] = br
    out[top - 1][right + 1] = bl
    out[bottom + 1][left - 1] = tr
    out[bottom + 1][right + 1] = tl
    return out
```

## Generator Constraints

ARC-GEN always emits a square 10x10 grid. The frame width and height are each 4..6. The top-left frame origin is at least one cell from every grid border, so all four outside outbox corners exist. The frame color is a random non-black color, and the four marker colors are distinct random colors excluding the frame color. The frame is hollow: its border is the frame color, its interior is black except for the four marker cells at the interior corners.

## Reference Notes

The ARC-DSL solution finds all colored objects, identifies the four size-one marker objects, identifies the largest object as the frame, takes the frame outbox corners, and paints each outbox corner with the farthest singleton marker by Manhattan distance. That farthest-marker rule is equivalent to the opposite-corner mapping above. The Code Golf 2025 solution rotates the grid four times and uses a regex substitution to move one marker per orientation, confirming the fourfold opposite-corner symmetry.
