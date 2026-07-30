# task260 Semantics

## Sources

- Current champion builder: `solutions_py/task260.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task260.json`
- ARC-GEN task id: `a78176bb`
- ARC-DSL task id: `a78176bb`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a78176bb.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a78176bb.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task260.py`

## Pattern

The input is a fixed 10x10 black grid with one colored diagonal line and one or two gray (`5`) wedge/corner regions touching one side of that diagonal. The non-gray, non-black color is the line color. The output removes all gray cells and draws extra full diagonals in the same line color, parallel to the original diagonal, just outside each gray region. The original colored diagonal remains.

Equivalently, every cell belongs to a diagonal indexed by `r - c`. Let `d0` be the diagonal index of the colored line. For each gray component, find the diagonal index `dg` of its relevant outside corner. If the gray component lies on the low-diagonal side of the colored line, draw the full diagonal `dg - 2`. If it lies on the high-diagonal side, draw the full diagonal `dg + 2`. Then turn all gray cells back to black.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    colors = {v for row in grid for v in row}
    line_color = next(v for v in colors if v not in (0, 5))

    out = [[0 for _ in range(n)] for _ in range(n)]
    colored_diags = set()
    gray_cells = []
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == line_color:
                colored_diags.add(r - c)
            elif v == 5:
                gray_cells.append((r, c))
    d0 = next(iter(colored_diags))

    target_diags = {d0}
    # The generator makes at most one gray wedge on each side. Classify gray cells
    # by whether their diagonal is below or above the colored diagonal and use the
    # extreme gray diagonal on that side to find the outside parallel line.
    low = [r - c for r, c in gray_cells if r - c < d0]
    high = [r - c for r, c in gray_cells if r - c > d0]
    if low:
        target_diags.add(min(low) - 2)
    if high:
        target_diags.add(max(high) + 2)

    for r in range(n):
        for c in range(n):
            if r - c in target_diags:
                out[r][c] = line_color
    return out
```

## Generator Constraints

The ARC-GEN generator uses fixed `size=10`. It chooses the colored diagonal index `diag` from `-5..5` and a random non-gray color. It samples one or two gray corner anchors from interior cells `1..8`, but keeps sampling until at least one valid anchor exists. A valid low-side anchor satisfies `row - col < diag - 1`, `col < size - diag`, and `row >= diag`; a valid high-side anchor satisfies `row - col > diag + 1`, `col >= -diag`, and `row < size + diag`. These constraints keep the generated gray wedge inside the 10x10 grid and separated from the colored diagonal by at least one diagonal.

The input draws the original colored diagonal. For each valid anchor, it fills a gray triangular/wedge region between the colored diagonal and the anchor diagonal, clipped by the anchor row/column quadrant. The output draws the original colored diagonal plus one parallel colored diagonal outside each gray wedge, offset by two diagonal units from the gray anchor diagonal. The output contains only black and the line color; all gray is removed.

## Reference Notes

The ARC-DSL solver identifies the unique nonzero, non-gray color, extracts gray objects, separates them by whether their upper-right corner is gray, then starts from either `urcorner + UP_RIGHT` or `llcorner + DOWN_LEFT`. From those starts it `shoot`s along both diagonal directions (`UNITY` and `NEG_UNITY`), fills with the line color, and finally replaces gray with black. This confirms the target line is a complete diagonal through the outside corner, not just the gray component boundary.

The Code Golf 2025 solution is a compact formula that uses diagonal-index tests around gray cells and color modulo behavior. It supports the same interpretation: detect whether exactly one gray witness lies at a diagonal offset and keep either existing non-gray/non-gray-modulo cells or the inferred parallel line color.
