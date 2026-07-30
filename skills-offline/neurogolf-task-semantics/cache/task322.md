# task322 Semantics

## Sources

- Current champion builder: `solutions_py/task322.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task322.json`
- ARC-GEN task id: `d037b0a7`
- ARC-DSL task id: `d037b0a7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d037b0a7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d037b0a7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task322.py`

## Pattern

The active grid is 3x3. Each column contains exactly one colored seed cell,
possibly color 0. The output keeps the seed color and fills the same color
straight downward to the bottom of that column. Cells above each seed remain
background/zero. The standard NeuroGolf tensor is padded to 30x30; only the
active top-left 3x3 region should contain output positives.

## Readable Python Solver

```python
def solve(grid):
    h = w = 3
    out = [[0 for _ in range(w)] for _ in range(h)]
    for c in range(w):
        seed_row = None
        seed_color = 0
        for r in range(h):
            if grid[r][c] != 0:
                seed_row = r
                seed_color = grid[r][c]
                break
        # ARC-GEN may choose color 0; then the visible input column is all zero.
        # In that case the output is also all zero for the column, so any row is equivalent.
        if seed_row is None:
            continue
        for r in range(seed_row, h):
            out[r][c] = seed_color
    return out
```

## Generator Constraints

ARC-GEN uses `size=3`. It samples three row coordinates independently in
`[0, 2]`, one for each column, and samples three distinct colors from
`range(10)`. If a sampled color is 0, the seed is visually background and the
output column remains zero. There is one seed per column by construction, no
extra objects, no variable dimensions in the bundled generator, and no
multi-cell seed objects.

## Reference Notes

The ARC-DSL solver extracts objects and paints a downward ray from each object's center using the object's own color. The Code Golf solution computes row-wise column maxima over prefixes, which works because each column has at most one nonzero color and the fill is downward. The generator clarifies the only ambiguous case: color 0 seeds are legal but produce an all-zero visible column.
