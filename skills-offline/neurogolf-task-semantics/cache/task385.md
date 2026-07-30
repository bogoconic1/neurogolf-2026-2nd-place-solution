# task385 Semantics

## Sources

- Current champion builder: `solutions_py/task385.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task385.json`
- ARC-GEN task id: `f25ffba3`
- ARC-DSL task id: `f25ffba3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f25ffba3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f25ffba3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task385.py`

## Pattern

The meaningful ARC grid is 10 rows by 4 columns in the top-left of the NeuroGolf canvas. Rows 0..4 of the input are blank padding and rows 5..9 contain colored vertical stacks rising from the bottom. The output preserves rows 5..9 and mirrors those five rows upward into rows 0..4: output row 0 is input row 9, row 1 is input row 8, row 2 is input row 7, row 3 is input row 6, row 4 is input row 5, and rows 5..9 are unchanged. All columns outside the four generated columns and all NeuroGolf canvas padding remain no-color.

## Readable Python Solver

```python
def solve(grid):
    bottom = grid[5:10]
    return bottom[::-1] + bottom
```

## Generator Constraints

ARC-GEN creates a width-4, height-10 grid. The lower half has four primary columns with lengths `[4..5, 2..5, 1..2, 0..2]`, plus optionally 0..2 extra short stacks in columns 2 and/or 3 with lengths 1..2. Stack colors are selected from four distinct random colors by an index list, so an extra stack may reuse a previous column color. The generator may flip the entire grid horizontally. The top five rows start blank and the output is always the lower five rows mirrored around the boundary between rows 4 and 5.

## Reference Notes

The ARC-DSL solver is exactly `vconcat(hmirror(bottomhalf(I)), bottomhalf(I))`; for grid rows this means reverse the lower half vertically, not a left-right horizontal mirror. The Code Golf 2025 solution is `g[:4:-1] + g[5:]`, matching the same row mapping. There is no color-specific tie-breaking; colors are simply copied from the lower-half rows.
