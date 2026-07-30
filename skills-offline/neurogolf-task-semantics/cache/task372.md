# task372 Semantics

## Sources

- Current champion builder: `solutions_py/task372.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task372.json`
- ARC-GEN task id: `e98196ab`
- ARC-DSL task id: `e98196ab`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e98196ab.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e98196ab.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task372.py`

## Pattern

The input is an 11-column by 11-row grid split into three bands: a 5-row top layer, one gray separator row, and a 5-row bottom layer. Colored pixels in the top layer and bottom layer are sparse. The output is a 5x11 grid that overlays the top layer onto the bottom layer at matching coordinates.

For generator-valid inputs the top and bottom colored positions come from one shared random pixel set with each pixel assigned to exactly one layer, so the two layers do not conflict at the same coordinate. A simple cellwise maximum or nonzero overlay is therefore equivalent to painting top objects onto the bottom half.

## Readable Python Solver

```python
def solve(grid):
    top = grid[:5]
    bottom = grid[6:11]
    return [[top[r][c] or bottom[r][c] for c in range(11)] for r in range(5)]
```

## Generator Constraints

- Width is fixed at 11 and logical half-height is fixed at 5.
- Input height is `2 * height + 1 = 11`.
- Row 5 is entirely gray separator color 5.
- A random subset of cells in a 5x11 coordinate grid is chosen with density about 0.2.
- Each selected coordinate is assigned index 0 or 1. Index 0 is placed in the top layer; index 1 is placed in the bottom layer.
- Two non-gray colors are sampled for the two layers.
- Because coordinates are selected once before layer assignment, no top/bottom coordinate conflict is generated.

## Reference Notes

ARC-DSL takes the top half and bottom half, extracts/merges top-half objects, and paints them onto the bottom half. Code Golf 2025 repeatedly pops the top row and takes `max(top_row, bottom_row)` against the corresponding bottom row, relying on the non-overlap/generator color constraints.
