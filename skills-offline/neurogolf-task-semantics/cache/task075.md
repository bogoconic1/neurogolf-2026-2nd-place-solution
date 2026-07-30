# task075 Semantics

## Sources

- Current champion builder: `solutions_py/task075.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task075.json`
- ARC-GEN task id: `363442ee`
- ARC-DSL task id: `363442ee`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_363442ee.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_363442ee.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task075.py`

## Pattern

The input is a `9x13` grid embedded in the standard NeuroGolf tensor. The
upper-left `3x3` block is a colored stamp. Column `3` is a vertical gray bar.
The right side is a `3x3` array of possible stamp slots; blue marker cells sit
at the centers/anchors of selected slots. The output preserves the stamp and
gray bar on the left, and paints a copy of the `3x3` stamp into each selected
right-side slot.

## Readable Python Solver

```python
def solve(grid):
    stamp = [row[:3] for row in grid[:3]]
    out = [row[:] for row in grid]
    for r in range(3):
        for c in range(3):
            marker_r = 3 * r + 1
            marker_c = 3 * (c + 1) + 2
            if grid[marker_r][marker_c] == 1:
                top = 3 * r
                left = 3 * (c + 1) + 1
                for sr in range(3):
                    for sc in range(3):
                        out[top + sr][left + sc] = stamp[sr][sc]
    return out
```

## Generator Constraints

- Stamp size is fixed at `3x3`.
- Actual grid size is fixed at `9x13` (`3 * size` rows and `4 * size + 1` cols).
- Column `3` is always gray (`5`) for all nine rows.
- The stamp colors are sampled from `3..5` non-blue, non-gray colors; therefore
  stamp cells never use blue (`1`) or gray (`5`) from the random color list.
- Blue marker cells use color `1` and occupy the centers/anchors of selected
  right-side `3x3` blocks: rows `1,4,7` and cols `5,8,11`.
- Between two and seven marker slots are selected.
- Output contains stamp copies in selected slots; unselected right-side slots
  remain black. The left stamp and gray bar remain visible.

## Reference Notes

ARC-DSL finds all color-1 marker cells, crops the top-left `3x3` stamp, converts it to an object, shifts the stamp object by each marker coordinate decremented by `(1,1)`, and paints all shifted copies onto the input.

The Code Golf 2025 solution computes each right-side output cell from the stamp cell at `(row mod 3, col mod 3)` gated by the corresponding marker block.
