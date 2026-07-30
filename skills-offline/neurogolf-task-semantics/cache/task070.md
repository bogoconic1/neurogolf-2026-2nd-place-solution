# task070 Semantics

## Sources

- Current champion builder: `solutions_py/task070.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task070.json`
- ARC-GEN task id: `32597951`
- ARC-DSL task id: `32597951`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_32597951.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_32597951.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task070.py`

## Pattern

The input is a 17x17 binary background pattern using black and blue. A hidden rectangular region is overlaid: cells in the rectangle that were not blue become cyan, while cells that were blue remain blue and are therefore hidden gaps in the cyan rectangle. The output preserves the grid, keeps the visible cyan cells, and changes exactly the hidden blue cells inside the cyan rectangle to green.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    cyan = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == 8]
    rows = [r for r, _ in cyan]
    cols = [c for _, c in cyan]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = [row[:] for row in grid]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if grid[r][c] == 1:
                out[r][c] = 3
    return out
```

## Generator Constraints

- Grid size is fixed at `17x17`.
- The base background is either totally random 17x17 or a periodic pattern with period width/height in `2..4`.
- The base colors are black (`0`) and blue (`1`).
- A rectangle with width and height in `2..10` is placed inside the grid.
- Cyan (`8`) is written on rectangle cells whose base value was not blue. Blue cells inside the rectangle stay blue in the input and become green (`3`) in the output.
- The generator rejects rectangles unless every rectangle row and every rectangle column has at least one visible cyan cell, so the cyan bounding box is the true rectangle bounds.

## Reference Notes

The ARC-DSL solver computes the set of cyan cells, takes its row/column span with `delta`, and fills that span with green over the input. Because the fill only affects cells in the rectangle that were not already cyan, the visible cyan boundary/interior remains cyan while hidden blue cells become green. The Code Golf solution compresses the same idea by using row/column membership in the cyan set.
