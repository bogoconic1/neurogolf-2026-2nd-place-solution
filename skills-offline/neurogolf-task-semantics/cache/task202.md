# task202 Semantics

## Sources

- Current champion builder: `solutions_py/task202.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task202.json`
- ARC-GEN task id: `855e0971`
- ARC-DSL task id: `855e0971`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_855e0971.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_855e0971.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task202.py`

## Pattern

The input is a rectangle split into contiguous same-color stripes. In the untransposed form the stripes are horizontal bands: each row has one non-black color, and contiguous rows of the same color form one stratum. Some cells inside the stripes are black marker pixels. The output preserves all stripe colors except that every black marker is expanded into a complete black line across the full thickness of its own stripe. For horizontal bands, a black marker at column `c` makes every row in that color band black at column `c`. If the generator transposes the example, the same rule appears as vertical color bands and each marker expands horizontally across the full width of its vertical band.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])

    def nz_colors(values):
        return {v for v in values if v != 0}

    def transpose(g):
        return [list(row) for row in zip(*g)]

    def solve_horizontal(g):
        h, w = len(g), len(g[0])
        out = [row[:] for row in g]
        row_color = []
        for row in g:
            colors = [v for v in row if v != 0]
            row_color.append(colors[0])

        start = 0
        while start < h:
            color = row_color[start]
            end = start + 1
            while end < h and row_color[end] == color:
                end += 1
            marked_cols = {
                c
                for r in range(start, end)
                for c, value in enumerate(g[r])
                if value == 0
            }
            for r in range(start, end):
                for c in marked_cols:
                    out[r][c] = 0
            start = end
        return out

    # In horizontal orientation each row contains at most one non-black color.
    horizontal = all(len(nz_colors(row)) <= 1 for row in grid)
    if horizontal:
        return solve_horizontal(grid)
    return transpose(solve_horizontal(transpose(grid)))
```

## Generator Constraints

ARC-GEN chooses width `10..20`, `2..5` distinct non-black stripe colors, and a height `2..10` for each stripe. For each stripe it places `0..2` black marker pixels at sampled row positions within that stripe; the generator repeats until there is at least one marker overall. Marker columns are sampled without replacement across all markers. The final grid is optionally transposed. Therefore examples may have horizontal or vertical stripes, dimensions up to roughly `20 x 50`, and one or more stripes may have no marker. Adjacent stripes have distinct colors because `random_colors` is used.

## Reference Notes

The ARC-DSL solver detects orientation using full color frontiers, mirrors/rotates into the horizontal-band case, isolates black-marker components, builds vertical frontiers from each marker inside its subgrid, intersects them with the same-color object, fills those indices with black, and mirrors back if needed. The Code Golf solution implements the same orientation-normalized cumulative stripe-marker idea in a compressed recursive form.
