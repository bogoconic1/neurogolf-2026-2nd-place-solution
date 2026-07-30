# task249 Semantics

## Sources

- Current champion builder: `solutions_py/task249.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task249.json`
- ARC-GEN task id: `a416b8f3`
- ARC-DSL task id: `a416b8f3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a416b8f3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a416b8f3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task249.py`

## Pattern

The output is the input grid duplicated horizontally. Every row is copied twice side by side: `output[r] = input[r] + input[r]`. The output height equals the input height, and the output width is exactly twice the input width. Colors and black background cells are preserved unchanged.

## Readable Python Solver

```python
def solve(grid):
    return [row + row for row in grid]
```

## Generator Constraints

ARC-GEN samples input width and height independently in `[3, 5]`. It creates a black grid of that size, samples a random set of occupied pixels, chooses a palette with up to about `width * height // 3` colors, and assigns sampled colors to those pixels. The generated output has the same height and width `2 * input_width`; each colored or black cell appears in both the left and right half at the same row and relative column.

## Reference Notes

The ARC-DSL solver is exactly `hconcat(I, I)`. The Code Golf 2025 solution is `lambda g: [r*2 for r in g]`, confirming there is no color remapping, object detection, cropping, or conditional behavior beyond horizontal duplication.
