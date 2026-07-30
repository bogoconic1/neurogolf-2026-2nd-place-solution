# task152 Semantics

## Sources

- Current champion builder: `solutions_py/task152.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task152.json`
- ARC-GEN task id: `67e8384a`
- ARC-DSL task id: `67e8384a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_67e8384a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_67e8384a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task152.py`

## Pattern

The input is a fixed `3x3` grid. Each input cell is chosen from a four-color palette, and colors may repeat in that palette. The output is a `6x6` grid made by mirroring the input horizontally and vertically: the top half is each input row followed by its left-right reverse, and the bottom half is the top half in reverse row order.

## Readable Python Solver

```python
def solve(grid):
    top = [row + row[::-1] for row in grid]
    return top + top[::-1]
```

## Generator Constraints

ARC-GEN always uses `size=3`. It samples four colors with replacement from all ARC colors, then samples nine indices into that four-color list. Because palette colors can repeat, distinct input labels are not guaranteed to map to distinct colors, but the transformation is purely geometric and color-preserving. Output shape is always `6x6`.

## Reference Notes

ARC-DSL computes `vmirror(I)`, concatenates it to the right of `I`, then vertically mirrors that 3x6 block and concatenates below it. The Code Golf 2025 solution is exactly `[r+r[::-1] for r in g+g[::-1]]`, confirming the quadrant mirror with no color changes.
