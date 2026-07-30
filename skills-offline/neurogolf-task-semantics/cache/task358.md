# task358 Semantics

## Sources

- Current champion builder: `solutions_py/task358.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task358.json`
- ARC-GEN task id: `e21d9049`
- ARC-DSL task id: `e21d9049`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e21d9049.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e21d9049.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task358.py`

## Pattern

The output is a full horizontal and vertical cross on a black background. The cross is centered at one row and one column. Colors follow a periodic sequence of length 3 or 4 using `(r + c) % period` along both arms. The input contains only a short sampled segment of that cross: `period` cells on the center row and `period` cells on the center column, shifted by an offset relative to the actual center. A horizontal flip may be applied to both input and output.

The task is to recover the center row, center column, period/color cycle, offset, and optional horizontal flip from the visible sampled cross, then extend the periodic colors across every cell in the matching center row or center column. All non-cross cells remain black.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, *r: [max(r[::6 ^ 83 >> len({*(r := ((0,) * 35 + r))})]) for _ in r] or [*map(p, g, *map(p, g, *g))]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Width is 10..20.
- Height is either equal to width or width + 1.
- Period/color count is 3 or 4.
- Center row and center column are at least `period` cells away from borders.
- Offset is 0..period-1 and shifts the visible sampled segment relative to the real center.
- The visible input has nonzero cells only on the sampled center row/column segment.
- Optional horizontal flip mirrors both input and output.
- Output shape matches input shape and is embedded in the standard NeuroGolf top-left canvas.

## Reference Notes

The ARC-DSL solver identifies the colored object, expands it by shifts related to its shape and neighbor offsets, then keeps only cells horizontally or vertically matching the recovered cross anchor. The Code Golf solution is recursive/golfed but matches the same periodic row/column extension. The ARC-GEN generator is the clearest source for period length, offset, center bounds, and the horizontal flip branch.
