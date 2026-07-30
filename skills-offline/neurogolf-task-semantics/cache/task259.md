# task259 Semantics

## Sources

- Current champion builder: `solutions_py/task259.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task259.json`
- ARC-GEN task id: `a740d043`
- ARC-DSL task id: `a740d043`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a740d043.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a740d043.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task259.py`

## Pattern

The input is a 5x5 to 7x7 grid filled with blue (`1`) background. A small colored sprite is placed at an arbitrary row/column offset. The sprite occupies a 2x2, 2x3, 3x2, or 3x3 bounding box and is generated from a hollow Conway-style binary shape. Its non-background pixels use two colors chosen from non-blue colors; colors may repeat across the sprite.

The output is the tight bounding box of the non-blue sprite, preserving all sprite colors and replacing any blue background cells inside that bounding box with black (`0`). In other words: crop the minimal rectangle containing all cells whose color is not blue, then map blue to black inside that crop.

## Readable Python Solver

```python
def solve(grid):
    rows = []
    cols = []
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color != 1:
                rows.append(r)
                cols.append(c)
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = []
    for r in range(r0, r1 + 1):
        out_row = []
        for c in range(c0, c1 + 1):
            color = grid[r][c]
            out_row.append(0 if color == 1 else color)
        out.append(out_row)
    return out
```

## Generator Constraints

Input width and height are independently chosen from `5..7`. The sprite bounding box width and height are independently chosen from `2..3`. The sprite row and column offsets keep the full bounding box inside the input grid. The binary sprite mask comes from `common.hollow_conway(wide, tall, variant)` and is nonempty; the output dimensions are exactly the generated sprite bounding box. The input background is always blue (`1`), while sprite cells use two random colors excluding blue. Blue cells inside the bounding box are holes/gaps and become black in the output.

Validation examples cover 2x2, 2x3, 3x2, and 3x3 output boxes, different offsets, repeated colors, and holes inside the bounding box.

## Reference Notes

The ARC-DSL solver finds all non-background objects, merges them, takes `subgrid` over the merged object, then replaces `ONE` with `ZERO`. That confirms the task is a tight crop over all non-blue pixels, not per-color extraction or shape normalization.

The Code Golf 2025 solution repeatedly transposes and trims rows/columns based on whether the border contains non-blue cells, then substitutes blue with black. This matches the tight-crop interpretation and explains why orientation is symmetric.
