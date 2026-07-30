# task318 Semantics

## Sources

- Current champion builder: `solutions_py/task318.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task318.json`
- ARC-GEN task id: `ce4f8723`
- ARC-DSL task id: `ce4f8723`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ce4f8723.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ce4f8723.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task318.py`

## Pattern

The input is a fixed 9x4 grid made from two 4x4 panels separated by a one-row yellow divider. The top panel contains color `1` pixels on background `0`; the bottom panel contains color `2` pixels on background `0`. The output is a 4x4 grid with color `3` everywhere except cells where both corresponding top and bottom panel cells are background `0`; those shared-empty cells become `0`. Equivalently, output `0` at the intersection of the zero cells of the two halves, and output `3` elsewhere.

## Readable Python Solver

```python
def solve(grid):
    h = 4
    out = [[3 for _ in range(4)] for _ in range(4)]
    top = grid[:h]
    bottom = grid[h + 1:h + 1 + h]
    for r in range(h):
        for c in range(4):
            if top[r][c] == 0 and bottom[r][c] == 0:
                out[r][c] = 0
    return out
```

## Generator Constraints

ARC-GEN always uses `size=4`, so input shape is `[9,4]` and output shape is `[4,4]`. Row `4` is all yellow divider cells. For color `1`, the generator samples a random subset of top-panel cells and paints them with `1`; for color `2`, it independently samples a random subset of bottom-panel cells and paints them with `2`. Background is `0`; output foreground is `3`. Public examples include varying sparse/dense subsets, but no variable size, no alternate colors, and no object geometry beyond per-cell occupancy.

## Reference Notes

The ARC-DSL solver takes `tophalf(I)` and `bottomhalf(I)`, computes `ofcolor(..., ZERO)` for both halves, intersects those coordinates, creates a 4x4 canvas of color `3`, and fills the intersection with `0`. The Code Golf solution encodes the same rule recursively/tersely: compare top cells against the corresponding bottom panel cells and emit `3` when they differ from the shared-zero condition. There is no tie-breaking or object identity logic.
