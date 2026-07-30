# task113 Semantics

## Sources

- Current champion builder: `solutions_py/task113.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task113.json`
- ARC-GEN task id: `496994bd`
- ARC-DSL task id: `496994bd`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_496994bd.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_496994bd.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task113.py`

## Pattern

The original ARC grid has height 10 and width 2..10. The top half consists of one to four colored horizontal rows. The output keeps the top five rows and mirrors them vertically into the bottom five rows. In NeuroGolf's padded 30x30 tensor, rows 10..29 are blank padding.

Readable row rule for the 30-row tensor: output row indices gather input rows `[0,1,2,3,4,4,3,2,1,0,10,10,...,10]`, with row 10 used as a guaranteed blank row for all padded output rows.

## Readable Python Solver

```python
def solve(grid):
    top = grid[:5]
    return top + top[::-1]
```

For the padded NeuroGolf frame, keep the first ten solved rows and leave rows 10..29 blank/background padding.

## Generator Constraints

- Width is randomly 2..10.
- Original ARC height is always 10 (`2 * height`, with `height=5`).
- The `colors` list has length 2..4, and each listed row color may repeat.
- Only the first `len(colors)` rows in the top half are non-background; the rest of the top five rows can be blank.
- The output mirrors the entire top five-row block, not just non-background rows.
- There are no dynamic object choices or tie-breaks.

## Reference Notes

- ARC-DSL crops the top half and vertically concatenates it with its horizontal mirror in DSL naming (`hmirror` mirrors row order).
- The Code Golf solution is exactly `g[:5] + g[4::-1]`.
