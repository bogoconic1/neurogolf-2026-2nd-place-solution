# task012 Semantics

## Sources

- Current champion builder: `solutions_py/task012.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task012.json`
- ARC-GEN task id: `0962bcdd`
- ARC-DSL task id: `0962bcdd`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_0962bcdd.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_0962bcdd.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task012.py`

## Pattern

The visible task grid is a fixed 12x12 square. There are two separated objects. Each object has a center pixel in the rare color `c1` and four orthogonal neighbor pixels in the arm color `c2`, forming a length-1 plus around the center. The output keeps the input pixels, extends each orthogonal arm to length 2 in color `c2`, and adds both diagonal lines through the center out to Chebyshev distance 2 in color `c1`. The result is an X plus an extended plus around each center.

The generator applies one of four gravity orientations to the whole grid, so the pair of objects can be seen in any of four rotated/reflected placements. The local rule around each center is the same after orientation.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0) + 1
    center_color = min(counts, key=counts.get)
    arm_color = max(counts, key=counts.get)

    out = [row[:] for row in grid]
    centers = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == center_color]
    for r, c in centers:
        for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            for k in (1, 2):
                rr, cc = r + k * dr, c + k * dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr][cc] = arm_color
        for dr, dc in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            for k in (1, 2):
                rr, cc = r + k * dr, c + k * dc
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr][cc] = center_color
    return out
```

## Generator Constraints

- Size is fixed at 12x12 before embedding into the NeuroGolf 30x30 tensor.
- Exactly two colors are chosen and they are distinct.
- Before gravity, centers are at rows 2 and 8, with columns independently chosen from `3..size-3` inclusive. This keeps all distance-2 arms inside the grid.
- Input has exactly two center-color pixels and eight arm-color pixels.
- Output has the same centers, arm color at orthogonal distance 1 and 2, and center color at diagonal distance 1 and 2.
- `gravity` is one of four orientations applied consistently to both input and output.

## Reference Notes

- ARC-DSL identifies the rare nonzero color as the center color, then finds the arm color after replacing background with the center color. It fills the arm color into direct neighbors of the arm pixels, then for each resulting object connects opposite corners to draw the two diagonals in the center color.
- ARC-GEN confirms that center color occurs twice; arm color occurs eight times in the input.
- The Code Golf solution encodes the same local completion by regex-like scans over flattened oriented grids.
