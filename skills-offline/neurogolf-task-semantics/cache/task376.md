# task376 Semantics

## Sources

- Current champion builder: `solutions_py/task376.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task376.json`
- ARC-GEN task id: `eb281b96`
- ARC-DSL task id: `eb281b96`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_eb281b96.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_eb281b96.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task376.py`

## Pattern

The input is a narrow, horizontally symmetric stripe pattern on a black background. It has width 17 and height `h = stretch + 2`, where `stretch` is 1..4 in the generator. One nonzero color is used. The output keeps the same width and expands the rows into a vertical repeated motif:

```python
output_rows = 2 * (input_rows + input_rows[1:-1]) + input_rows[:1]
```

Equivalently, for generator-valid inputs, copy the whole input, append its interior rows excluding the first and last, repeat that block twice, and finish with the original first row. The output height is `2 * (h + h - 2) + 1`, i.e. `4 * h - 3`, which matches `5 + 4 * stretch`.

## Readable Python Solver

```python
def solve(grid):
    rows = [list(row) for row in grid]
    return [list(row) for row in (rows + rows[1:-1]) * 2 + rows[:1]]
```

## Generator Constraints

- `stretch` is randomly chosen from 1..4 when not fixed.
- `color` is one random nonzero ARC color.
- Width defaults to 17 and is not randomized by `validate`.
- Input height is `2 + stretch`, so known/generated heights are 3..6.
- Output height is `5 + 4 * stretch`, so output heights are 9, 13, 17, or 21.
- Columns with `c % 4 == 2` have the color in the top row and corresponding repeated rows in the output.
- Columns with `c % 4 == 0` have the color in the bottom input row and its repeated counterpart.
- Columns with odd `c` contain the vertical middle stretch in the input and the repeated middle runs in the output.
- The generated motif is horizontally symmetric for width 17, so the ARC-DSL horizontal mirror is a no-op on generator-valid rows.

## Reference Notes

ARC-DSL constructs the output by concatenating the input with a horizontal mirror of the input without its last row, then concatenating that result with itself minus the first row. Because the generated rows are horizontally symmetric, this is equivalent to repeating `input + input[1:-1]` and appending the first row. The Code Golf 2025 solution expresses exactly that compact generator-specific form:

```python p = lambda g: 2 * (g + g[1:-1]) + g[:1] ```

There is no color inference beyond copying rows; the nonzero color can be any ARC color, and black remains black.
