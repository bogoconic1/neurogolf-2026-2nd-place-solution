# task197 Semantics

## Sources

- Current champion builder: `solutions_py/task197.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task197.json`
- ARC-GEN task id: `82819916`
- ARC-DSL task id: `82819916`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_82819916.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_82819916.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task197.py`

## Pattern

The input is an even-width, even-height grid containing several separated colored rows on black background. Row 1 is the first colored row and contains the complete two-color pattern across the whole width. Later colored rows use the same binary pattern but a different pair of colors. In the input, each later colored row shows only the prefix through the first occurrence of both row colors; the rest of that row is black. The output fills each colored row across the full width by substituting that row's two colors into the binary pattern learned from row 1.

Black rows remain black. Existing visible prefix cells remain the same; only missing suffix cells are filled.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    pattern_row = grid[1]
    first_color = pattern_row[0]
    try:
        first_other_col = next(c for c, v in enumerate(pattern_row) if v != first_color)
    except StopIteration:
        return [row[:] for row in grid]

    out = [row[:] for row in grid]
    for r, row in enumerate(grid):
        if not any(row):
            continue
        a = row[0]
        b = row[first_other_col]
        for c, pv in enumerate(pattern_row):
            out[r][c] = a if pv == first_color else b
    return out
```

This equivalent implementation assumes the generator invariant that every later visible prefix includes both row colors and therefore includes the column of the first non-row1 color.

## Generator Constraints

- Width is `2 * randint(4,5)`, so width is 8 or 10.
- Height is `2 * randint(3,7)`, so height is 6, 8, 10, 12, or 14.
- The binary pattern has both values 0 and 1.
- Colored rows start at row 1, then advance by a random step 2..4 until below height.
- Each colored row uses two random ARC colors.
- The first colored row is fully visible in the input.
- Later colored rows reveal cells from the left until both row colors have appeared; cells after that prefix are black in the input but filled in the output.
- Output size equals input size; NeuroGolf output remains the dense `[1,10,30,30]` canvas.

## Reference Notes

The ARC-DSL solver identifies the largest object as the full pattern row, normalizes it, determines the two pattern classes from the first and other color positions, then recolors/paints all other row objects according to the normalized pattern.

The Code Golf 2025 solution is `[[r[g[1].index(v)] for v in g[1]] for r in g]`: for each row, use the full row `g[1]` as the pattern and map every pattern value to the corresponding visible color in the current row at the first occurrence of that pattern value.
