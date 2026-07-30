# task067 Semantics

## Sources

- Current champion builder: `solutions_py/task067.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task067.json`
- ARC-GEN task id: `2dee498d`
- ARC-DSL task id: `2dee498d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_2dee498d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_2dee498d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task067.py`

## Pattern

The input is a `size x (3*size)` grid made of three side-by-side square panels. The left and right panels are identical. The middle panel is the same square, either unchanged or vertically flipped. The output is the leftmost `size x size` panel.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    return [row[:n] for row in grid]
```

## Generator Constraints

- `size` is sampled from `2..5` in ARC-GEN; NeuroGolf tensors are padded to `30x30`.
- The source square uses a sampled palette of 2 to 4 colors, with arbitrary per-cell choices from that palette.
- The input width is exactly `3*size` before padding.
- The right panel equals the left panel. The middle panel equals the left panel or its vertical flip.
- Output size is `size x size` and is exactly the first third of the input.

## Reference Notes

The ARC-DSL solver is simply `first(hsplit(I, THREE))`. The Code Golf solution uses the same fact: for each row, keep the first `len(grid)` columns. The middle panel's optional flip is irrelevant because the left panel is always already correct.
