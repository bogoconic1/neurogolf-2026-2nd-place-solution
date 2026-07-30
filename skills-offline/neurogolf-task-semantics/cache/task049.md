# task049 Semantics

## Sources

- Current champion builder: `solutions_py/task049.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task049.json`
- ARC-GEN task id: `23b5c85d`
- ARC-DSL task id: `23b5c85d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_23b5c85d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_23b5c85d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task049.py`

## Pattern

The input contains two to five solid, axis-aligned colored rectangles on a
background. The desired object is the smallest visible rectangle, which is also
guaranteed by ARC-GEN to have the rarest color after all rectangles are drawn.
The output is the tight subgrid of that smallest object: a solid rectangle whose
height and width match the visible smallest rectangle and whose cells all have
that object's color.

## Readable Python Solver

```python
def solve(grid):
    counts = {}
    for row in grid:
        for v in row:
            if v:
                counts[v] = counts.get(v, 0) + 1
    color = min(counts, key=counts.get)
    rows = [row for row in grid if color in row]
    height = len(rows)
    width = max(row.count(color) for row in rows)
    return [[color for _ in range(width)] for _ in range(height)]
```

## Generator Constraints

ARC-GEN samples input width and height from 10..20 and draws 2..5 rectangles.
Rectangle widths and heights are sorted descending before placement, and the
last rectangle is forced to be strictly smaller in both width and height than
the previous rectangle. The last rectangle is drawn last, so it remains visible
as a solid rectangle. The generator rejects cases where that last rectangle's
visible color is not strictly the rarest foreground color. Hand examples cover
up to 20x20 inputs and outputs as large as roughly 3x3.

## Reference Notes

The ARC-DSL solver extracts foreground objects, chooses the object with minimal size, and returns its tight subgrid. The Code Golf solution finds the rarest foreground color and reconstructs the solid rectangle by counting that color in input rows. These agree because the generator makes the smallest object the rarest visible color.
