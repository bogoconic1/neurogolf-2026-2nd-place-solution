# task142 Semantics

## Sources

- Current champion builder: `solutions_py/task142.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task142.json`
- ARC-GEN task id: `62c24649`
- ARC-DSL task id: `62c24649`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_62c24649.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_62c24649.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task142.py`

## Pattern

The input is a 3x3 grid with colors from `0..3`. The output is a 6x6 grid formed by mirroring the input horizontally and vertically: first concatenate each row with its reverse, then concatenate the resulting 3 rows with their vertical reverse. NeuroGolf pads this 6x6 result into the standard `[1,10,30,30]` output tensor.

## Readable Python Solver

```python
def solve(grid):
    top = [row + row[::-1] for row in grid]
    return top + top[::-1]
```

## Generator Constraints

- Input shape is always exactly `3x3`.
- Output shape is always exactly `6x6` before NeuroGolf padding.
- Input colors are sampled independently from `0..3`.
- No object detection, tie-breaking, cropping, or dynamic size inference is required.

## Reference Notes

The ARC-DSL solver computes `vmirror(I)`, horizontally concatenates it with the input, mirrors the result vertically, then vertically concatenates. The Code Golf solution `p=lambda g:[r+r[::-1]for r in g+g[::-1]]` is the same transformation.
