# task035 Semantics

## Sources

- Current champion builder: `solutions_py/task035.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task035.json`
- ARC-GEN task id: `1f642eb9`
- ARC-DSL task id: `1f642eb9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_1f642eb9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_1f642eb9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task035.py`

## Pattern

The input is a 10x10 grid containing one cyan (`8`) axis-aligned filled rectangle and zero or more single-cell non-cyan colored markers on the outer frame. The output keeps the input unchanged, then copies each marker onto the nearest cell on the cyan rectangle boundary along the same row or column:

- markers on row `0` are copied down to the rectangle top row;
- markers on row `9` are copied up to the rectangle bottom row;
- markers on column `0` are copied right to the rectangle left column;
- markers on column `9` are copied left to the rectangle right column.

If the marker aligns with a rectangle corner, the generated data has only the chosen source marker for that boundary cell. The copied marker color overwrites cyan at the projected rectangle boundary cell. Existing marker cells outside the rectangle remain in place.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]

    cyan = 8
    rows = [r for r, row in enumerate(grid) if cyan in row]
    cols = [c for row in grid for c, v in enumerate(row) if v == cyan]
    top, bottom = min(rows), max(rows)
    left, right = min(cols), max(cols)

    for c, color in enumerate(grid[0]):
        if color not in (0, cyan):
            out[top][c] = color
    for c, color in enumerate(grid[9]):
        if color not in (0, cyan):
            out[bottom][c] = color
    for r, row in enumerate(grid):
        color = row[0]
        if color not in (0, cyan):
            out[r][left] = color
    for r, row in enumerate(grid):
        color = row[9]
        if color not in (0, cyan):
            out[r][right] = color
    return out
```

## Generator Constraints

ARC-GEN always uses a 10x10 grid. The cyan rectangle has width `2..4`, height `2..5`, top row `3`, and right column `5`; therefore its columns are `6 - width .. 5` and rows are `3 .. 2 + height`. The rectangle is filled with cyan. For each rectangle boundary cell, the generator may add one marker at the corresponding outside-frame coordinate: top boundary to row `0`, bottom boundary to row `9`, left boundary to column `0`, and right boundary to column `9`. Corner boundary cells randomly choose one of their two possible outside-frame locations when a marker is emitted. Marker colors are random colors excluding cyan; background is zero. There may be no marker for some boundary cells, and multiple markers can share the same color.

## Reference Notes

The ARC-DSL solution separates singleton colored objects from the larger cyan rectangle, computes the gravitation direction from each singleton toward the rectangle, shifts the singleton by the one-step-corrected vector, and paints the shifted singleton into the original input. This matches the generator: markers move straight inward until they touch the rectangle edge. The Code Golf 2025 recursive zip solution implements the same pull-edge-markers-into-the-pool logic by repeatedly orienting the grid. There is no disagreement between the references.
