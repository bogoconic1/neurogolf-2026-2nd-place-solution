# task036 Semantics

## Sources

- Current champion builder: `solutions_py/task036.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task036.json`
- ARC-GEN task id: `1f85a75f`
- ARC-DSL task id: `1f85a75f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1f85a75f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1f85a75f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task036.py`

## Pattern

The input is a 30x30 noisy grid. One connected compact object in a single target color is embedded away from the border. Other colored pixels are sparse background noise in non-target colors. The output is the tight bounding-box crop of the target object, preserving the object's color and any background holes inside the box. Noise outside the target object's box is discarded.

The target object is the largest connected non-background object. In ARC-GEN it is also the only object using `colors[0]`; all later noise pixels use other colors and are prevented from landing in or immediately around the object's bounding box.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    seen = [[False] * w for _ in range(h)]
    best = []

    for r in range(h):
        for c in range(w):
            color = grid[r][c]
            if color == 0 or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            if len(comp) > len(best):
                best = comp

    r0 = min(r for r, _ in best)
    r1 = max(r for r, _ in best)
    c0 = min(c for _, c in best)
    c1 = max(c for _, c in best)
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]
```

## Generator Constraints

ARC-GEN always uses a 30x30 grid. The target object's bounding rectangle has width and height in `3..5`, and the object pixels are a connected random subset of that rectangle with about 75 percent density. The target object's top-left corner is chosen with at least five cells of margin from every grid edge. The target color is `colors[0]`; there are `2..4` total colors. Sparse noise uses only `colors[1:]` with density `(len(colors)-1)/50`, and generator code rejects noise in the one-cell halo around the target object's bounding rectangle, so the crop area contains only target pixels and background holes. Output dimensions are at most 5x5.

## Reference Notes

The ARC-DSL solver computes all objects, chooses the largest by size, and returns `subgrid(largest_object, I)`. This confirms that connectedness and largest-object selection are the core semantics, not color id or absolute location. The Code Golf 2025 solution is a compact recursive crop/filter expression that also isolates the main non-noise shape. There is no disagreement between references; ARC-GEN adds the useful bounds that the object crop is no larger than 5x5 and is well inside the 30x30 input.
