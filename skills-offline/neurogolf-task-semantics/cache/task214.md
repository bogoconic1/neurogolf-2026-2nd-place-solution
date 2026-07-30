# task214 Semantics

## Sources

- Current champion builder: `solutions_py/task214.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task214.json`
- ARC-GEN task id: `8e5a5113`
- ARC-DSL task id: `8e5a5113`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8e5a5113.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8e5a5113.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task214.py`

## Pattern

The input is a 3-row by 11-column grid. The left 3x3 block is filled with colors drawn from three non-gray colors. Columns 3 and 7 are gray separators. The rest of the input to the right is initially empty. The output preserves the left 3x3 block and separators, paints a 90-degree clockwise rotation of the 3x3 block between the separators, and paints a 180-degree rotation of the 3x3 block after the second separator.

In coordinates for size 3:

- output rows 0..2, cols 0..2 are the original tile.
- output col 3 and col 7 are gray separators.
- output[row=c][col=6-r] receives input[r][c] for the 90-degree rotation.
- output[row=2-r][col=10-c] receives input[r][c] for the 180-degree rotation.

## Readable Python Solver

```python
def solve(grid):
    tile = [row[:3] for row in grid[:3]]
    out = [row[:] for row in grid]
    for r in range(3):
        for c in range(3):
            out[c][6 - r] = tile[r][c]
            out[2 - r][10 - c] = tile[r][c]
    return out
```

## Generator Constraints

- The generated task uses `size=3` in all official examples.
- The dense grid has height 3 and width `3*size + 2 = 11`.
- Columns 3 and 7 are gray separators (color 5).
- The left 3x3 tile colors are sampled from three random non-gray colors.
- Output dimensions match the input dimensions before NeuroGolf padding to `[1, 10, 30, 30]`.

## Reference Notes

ARC-DSL crops the top-left 3x3 tile, computes `rot90` and `rot180`, shifts those objects by column offsets 4 and 8, and paints them onto the input. The Code Golf solution is a compact row/zip construction of the same two rotations.
