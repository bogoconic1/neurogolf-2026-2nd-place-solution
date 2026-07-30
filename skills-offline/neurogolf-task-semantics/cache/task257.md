# task257 Semantics

## Sources

- Current champion builder: `solutions_py/task257.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task257.json`
- ARC-GEN task id: `a68b268e`
- ARC-DSL task id: `a68b268e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a68b268e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a68b268e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task257.py`

## Pattern

The input is a 9x9 grid divided by a blue (`1`) center row and center column into four 4x4 quadrants. The top-left quadrant contains color `7`, the top-right contains color `4`, the bottom-left contains color `8`, and the bottom-right contains color `6`, all on black background.

The output is a 4x4 overlay of the four quadrants in their local coordinates. The bottom-right quadrant (`6`) is the base. Positions marked in the bottom-left quadrant become `8`, positions marked in the top-right quadrant become `4`, and positions marked in the top-left quadrant become `7`. Priority is `7 > 4 > 8 > 6 > 0`: if multiple quadrants have a colored cell at the same local coordinate, the higher-priority color is output.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(4)] for _ in range(4)]
    panels = [
        (5, 5, 6),  # bottom-right base
        (5, 0, 8),  # bottom-left overlay
        (0, 5, 4),  # top-right overlay
        (0, 0, 7),  # top-left overlay
    ]
    for r0, c0, color in panels:
        for r in range(4):
            for c in range(4):
                if grid[r0 + r][c0 + c] == color:
                    out[r][c] = color
    return out
```

## Generator Constraints

The generator uses fixed `size=4`, so every input is `2*size+1 = 9` square and every output is 4x4. The center row and center column are always blue separators. For each of four colors `(7, 4, 8, 6)`, a random set of local 4x4 pixel coordinates is generated and written into its quadrant using offsets `(top-left, top-right, bottom-left, bottom-right)`. Each quadrant may contain multiple colored cells; overlaps across quadrants happen only after projecting to local output coordinates.

Only colors `0,1,4,6,7,8` appear in the input. The output contains only `0,4,6,7,8`, with no blue separator. The priority order is fixed by the generator's reverse loop and is confirmed by the ARC-DSL solver: start from bottom-right `6`, fill `8`, then `4`, then `7`.

## Reference Notes

The ARC-DSL solver splits the input into top/bottom halves and then left/right halves, extracts positions of `4` from the top-right, `7` from the top-left, and `8` from the bottom-left, then fills those positions over the bottom-right panel. This confirms a local-coordinate overlay, not a geometric move or crop.

The Code Golf 2025 solution recursively doubles/transposes/list-pops to peel the 9x9 grid down to the projected 4x4 overlay. Its behavior matches the same quadrant-priority rule, but the ARC-GEN and ARC-DSL references are clearer for ONNX design.
