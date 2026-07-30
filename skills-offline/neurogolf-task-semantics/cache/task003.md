# task003 Semantics

## Sources

- Current champion builder: `solutions_py/task003.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task003.json`
- ARC-GEN task id: `017c7c7b`
- ARC-DSL task id: `017c7c7b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_017c7c7b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_017c7c7b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task003.py`

## Pattern

The input is a fixed 6x3 grid using black `0` and blue `1`. The output is a 9x3 grid. It preserves the six input rows at the top, appends three more rows that continue the vertical stencil pattern, and recolors every blue `1` in the full 9x3 result to red `2`. If the top half of the input (rows 0..2) is exactly equal to the bottom half (rows 3..5), append the bottom half. Otherwise append the 3x3 crop from rows 2..4. All black cells remain black.

## Readable Python Solver

```python
def solve(grid):
    top = [row[:] for row in grid[:3]]
    bottom = [row[:] for row in grid[3:6]]
    extra = bottom if top == bottom else [row[:] for row in grid[2:5]]
    return [[2 if cell == 1 else cell for cell in row] for row in grid + extra]
```

## Generator Constraints

ARC-GEN always uses width 3, input height 6, and output height 9. It chooses a period `steps` of 2 or 3. A small blue stencil is sampled inside one period: for period 2 exactly 3 blue pixels are selected from the 3x2 tile; for period 3 either 4 or 5 blue pixels are selected from the 3x3 tile. The stencil is repeated down the 9-row output. Period-2 cases may flip the stencil horizontally on alternating periods; period-3 cases do not flip. The input is the first six rows of this generated output but still colored blue `1`; the output is the first nine rows recolored to red `2`.

## Reference Notes

The ARC-DSL solver computes `tophalf(I)` and `bottomhalf(I)`, tests equality, chooses either `bottomhalf(I)` or `crop(I, (2, 0), (3, 3))`, concatenates that below the input, then replaces color `1` with color `2`. The Code Golf solution expresses the same rule compactly by appending one of the existing three-row slices and doubling nonzero cells. There is no ambiguity in the references: the only branch is whether rows 0..2 equal rows 3..5.
