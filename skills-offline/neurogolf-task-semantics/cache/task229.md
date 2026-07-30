# task229 Semantics

## Sources

- Current champion builder: `solutions_py/task229.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task229.json`
- ARC-GEN task id: `9565186b`
- ARC-DSL task id: `9565186b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_9565186b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_9565186b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task229.py`

## Pattern

The input is a 3x3 grid using two to four non-gray colors. Exactly one color is the unique majority/mode. The output keeps cells whose color is the majority color and replaces every other cell with gray `5`.

## Readable Python Solver

```python
def solve(grid):
    flat = [v for row in grid for v in row]
    mode = max(set(flat), key=flat.count)
    return [[v if v == mode else 5 for v in row] for row in grid]
```

## Generator Constraints

- Grid size is fixed at 3x3.
- Colors are sampled from two to four random colors excluding gray `5`.
- The generated color list is constructed with a unique maximum-count color.
- The output shape is the same 3x3 crop; NeuroGolf padding outside the crop must remain zero-hot.

## Reference Notes

- ARC-DSL selects the largest object/color class and paints it on a gray canvas.
- The Code Golf 2025 solution flattens the grid, computes the mode, and maps non-mode cells to gray.
