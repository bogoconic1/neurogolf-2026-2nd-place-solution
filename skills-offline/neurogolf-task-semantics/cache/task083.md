# task083 Semantics

## Sources

- Current champion builder: `solutions_py/task083.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task083.json`
- ARC-GEN task id: `3af2c5a8`
- ARC-DSL task id: `3af2c5a8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_3af2c5a8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_3af2c5a8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task083.py`

## Pattern

The input is a 3x4 grid with black background and one randomly chosen foreground color. The output mirrors that 3x4 image horizontally and vertically into a 6x8 image. Each foreground pixel at `(r, c)` is copied to `(r, c)`, `(5-r, c)`, `(r, 7-c)`, and `(5-r, 7-c)`. Background remains black.

There is no tie-breaking or color selection beyond preserving the single foreground color.

## Readable Python Solver

```python
def solve(grid):
    top = [row + row[::-1] for row in grid]
    return top + top[::-1]
```

## Generator Constraints

ARC-GEN uses fixed `height=3` and `width=4` for the folded input. It samples a random non-background color and a random subset of cells in the 3x4 input. The output size is exactly `2*height` by `2*width`, i.e. 6x8. Input pixels outside the 3x4 source area are zero-hot in NeuroGolf tensors.

## Reference Notes

ARC-DSL computes `vmirror(I)`, concatenates the input with the vertical mirror horizontally, then mirrors that horizontally and concatenates vertically. The Code Golf solution is the direct Python equivalent: for every row, append its reverse, then append the reverse of the row list.
