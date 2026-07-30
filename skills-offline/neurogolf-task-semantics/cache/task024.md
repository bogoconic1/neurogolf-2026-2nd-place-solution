# task024 Semantics

## Sources

- Current champion builder: `solutions_py/task024.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task024.json`
- ARC-GEN task id: `178fcbfb`
- ARC-DSL task id: `178fcbfb`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_178fcbfb.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_178fcbfb.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task024.py`

## Pattern

The input is a sparse rectangular grid containing colored point markers on background zero. Red markers (`2`) draw full-height vertical red lines through their columns. Blue (`1`) and green (`3`) markers draw full-width horizontal lines through their rows, in their own color. Horizontal blue/green rows override red vertical columns at intersections. The output has the same active height and width as the input.

Equivalently, for each output cell `(r, c)`: if input row `r` contains a blue or green marker, output that row color everywhere; otherwise, if input column `c` contains any red marker, output red; otherwise output zero.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    row_color = [0] * h
    red_cols = set()
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == 2:
                red_cols.add(c)
            elif v in (1, 3):
                row_color[r] = v
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r in range(h):
        if row_color[r]:
            out[r] = [row_color[r] for _ in range(w)]
        else:
            for c in red_cols:
                out[r][c] = 2
    return out
```

## Generator Constraints

- Active width and height are each sampled from 6 through 15.
- Marker colors are generated in order: one or two reds (`2`), one or two blues (`1`), and one or two greens (`3`). Thus there are 3 through 6 total markers.
- Marker rows are sampled without replacement from the active height, so no row contains two markers. Marker columns are sampled independently and may repeat.
- Red markers draw vertical lines. Blue and green markers draw horizontal lines. Because the generator iterates reds first and then blue/green markers, blue/green horizontal rows overwrite red vertical intersections.

## Reference Notes

- ARC-DSL builds red vertical frontiers from all red cells, fills them, then paints horizontal frontiers for every non-red object using that object's color.
- The Code Golf 2025 solution compresses the rule per cell: a row with max color 1 emits 1, a row with max color 3 emits 3, while rows with max 0 or 2 fall back to whether the column contains a red marker.
- Rows are unique across all markers, so the row maximum is a safe row-color selector for this generator.
