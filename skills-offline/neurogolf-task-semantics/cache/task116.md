# task116 Semantics

## Sources

- Current champion builder: `solutions_py/task116.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task116.json`
- ARC-GEN task id: `4c4377d9`
- ARC-DSL task id: `4c4377d9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_4c4377d9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_4c4377d9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task116.py`

## Pattern

The input is a small 3-row by 4-column grid with a background color and one foreground color. The output has the same width and double the height: the input is mirrored vertically above itself. Equivalently, output rows are input rows `[2, 1, 0, 0, 1, 2]`. No colors change, and all padded NeuroGolf rows/columns outside the ARC grid remain zero-hot.

## Readable Python Solver

```python
def solve(grid):
    return grid[::-1] + [row[:] for row in grid]
```

## Generator Constraints

- ARC-GEN uses fixed default dimensions `height=3`, `width=4` for random cases.
- The grid background is one random color and the marked pixels are a second random color.
- A random nonempty set of pixels is chosen in the 3x4 input; those cells are foreground.
- The output is always 6x4: rows 0..2 are the vertical mirror of input rows 2..0, and rows 3..5 are the original input rows 0..2.
- In NeuroGolf's padded `[1,10,30,30]` tensor, input row 3 and beyond are zero-hot, not ARC background color.

## Reference Notes

- ARC-DSL expresses the rule as `hmirror(I)` followed by `vconcat(x1, I)`.
- Code Golf 2025 is the same rule in Python: `g[::-1] + g`.
