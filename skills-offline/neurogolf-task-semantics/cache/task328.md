# task328 Semantics

## Sources

- Current champion builder: `solutions_py/task328.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task328.json`
- ARC-GEN task id: `d22278a0`
- ARC-DSL task id: `d22278a0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d22278a0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d22278a0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task328.py`

## Pattern

The input is a square grid with two to four colored marker pixels placed at
corners. The output colors cells according to a corner Voronoi rule. For each
cell, compute Manhattan distance to every active corner marker. If there is a
unique nearest active corner and the Chebyshev distance from that corner to the
cell is even, color the cell with that corner marker's color. If the nearest
corner is tied or the Chebyshev distance is odd, the cell remains black (`0`).
The original corner cells are included because their Chebyshev distance is zero.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    corners = []
    for r, c in ((0, 0), (0, n - 1), (n - 1, 0), (n - 1, n - 1)):
        if grid[r][c] != 0:
            corners.append((r, c, grid[r][c]))
    out = [[0 for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for c in range(n):
            dists = [abs(rr - r) + abs(cc - c) for rr, cc, _ in corners]
            best = min(dists)
            if dists.count(best) != 1:
                continue
            rr, cc, color = corners[dists.index(best)]
            if max(abs(rr - r), abs(cc - c)) % 2 == 0:
                out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN chooses a square size from 6 to 18. It samples 2 to 4 of the four corner
positions and assigns each active corner a random non-black color. Only corner
cells are colored in the input. The output uses the full square size; outside the
standard NeuroGolf dense canvas remains unsupported.

## Reference Notes

The ARC-DSL solver expresses the same rule by filtering cells that are uniquely closest to an object/corner under Manhattan distance and then applying an even Chebyshev-parity predicate relative to the winning corner. The Code Golf 2025 solution packs the same calculation into a sort key combining tie/nearest Manhattan distance and Chebyshev parity. References agree.
