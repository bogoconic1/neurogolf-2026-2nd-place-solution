# task361 Semantics

## Sources

- Current champion builder: `solutions_py/task361.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task361.json`
- ARC-GEN task id: `e40b9e2f`
- ARC-DSL task id: `e40b9e2f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e40b9e2f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e40b9e2f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task361.py`

## Pattern

The input is a 10x10 black grid containing a partial colored pinwheel around a central 2x2 or 3x3 square. Some pixels are present in all four rotated quadrants, while other pixels appear in only one quadrant. The output completes the pinwheel by rotating every visible local pixel around the center into all four quadrants, preserving each pixel's color.

The center can be detected from the dense all-quadrant core. For length 2 examples the generator uses `bump=1`; for length 3 or 4 it uses `bump=0`. The completed object is the union of the original local pattern and its 90/180/270 degree rotations around the detected center.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    pts = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c]]

    # Detect the dense 2x2 or 3x3 center block. Prefer 3x3 when present.
    center = None
    for k in (3, 2):
        for r in range(h - k + 1):
            for c in range(w - k + 1):
                if all(grid[rr][cc] for rr in range(r, r + k) for cc in range(c, c + k)):
                    # Use doubled center coordinates to avoid fractions.
                    center = (2 * r + k - 1, 2 * c + k - 1)
        if center is not None:
            break

    cr2, cc2 = center
    out = [row[:] for row in grid]
    for r, c, color in pts:
        dr2 = 2 * r - cr2
        dc2 = 2 * c - cc2
        for _ in range(3):
            dr2, dc2 = -dc2, dr2
            rr = (cr2 + dr2) // 2
            cc = (cc2 + dc2) // 2
            if 0 <= rr < h and 0 <= cc < w:
                out[rr][cc] = color
    return out
```

## Generator Constraints

- Grid size is always 10x10.
- The base local pattern length is 2..4. Length 2 uses `bump=True`; lengths 3 and 4 use `bump=False`.
- The center row/column are chosen so every generated rotated point stays in bounds.
- The local seed coordinates satisfy `0 <= r <= c < length`, with optional random omissions outside the dense core.
- Core pixels are placed in all quadrants (`idx=-1`); outer pixels may be placed in only one quadrant (`idx` in 0..3).
- Two nonzero colors are sampled, and every seed pixel chooses one of those colors.
- Output paints every seed pixel into all four rotated positions.

## Reference Notes

The ARC-DSL solver finds the first object, searches neighboring shifts of mirrored versions to maximize overlap, paints the best 180-degree completion, then repeats with a diagonal/vertical mirror composition to complete the remaining rotations.

The Code Golf 2025 solution scans for the dense 2x2/3x3 center, then for each nonzero cell repeatedly applies a 90-degree rotation in doubled-center coordinates and paints the resulting cells.
