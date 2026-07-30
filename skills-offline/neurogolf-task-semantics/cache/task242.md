# task242 Semantics

## Sources

- Current champion builder: `solutions_py/task242.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task242.json`
- ARC-GEN task id: `9ecd008a`
- ARC-DSL task id: `9ecd008a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_9ecd008a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_9ecd008a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task242.py`

## Pattern

The input is a 16x16 horizontally and vertically mirrored color grid with one missing 3x3 block replaced by black zeros. The missing block may appear in any mirrored quadrant in the fixed examples. The output is exactly the original 3x3 block. Because each row is horizontally mirrored, each missing row can be recovered from the same row on the opposite side: if the three zeros in a row start at column c, output that row as input columns 15-c, 14-c, 13-c in that order. Keep only the three rows that contain zeros, ordered top to bottom.

## Readable Python Solver

```python
def solve(grid):
    out = []
    for row in grid:
        if 0 not in row:
            continue
        c0 = row.index(0)
        out.append([row[15 - c0 - dc] for dc in range(3)])
    return out
```

## Generator Constraints

ARC-GEN builds an 8x8 random color bitmap, mirrors it across both axes into a 16x16 grid, then blanks a 3x3 cutout and returns the removed values as the 3x3 output. Random generation uses colors 1..9 and size=8, minisize=3; explicit train/test fixtures include cutouts outside the random branch top-left range, including bottom-half and right-half locations. The blanked block is always exactly three contiguous zero columns across exactly three contiguous rows. No other zeros are generated in the colored mirrored grid.

## Reference Notes

ARC-DSL solves this by vertically mirroring the input and taking the subgrid at the zero-cell coordinates; because the full grid is symmetric in both axes, horizontal mirroring of each zero row gives the same missing values. The Code Golf 2025 one-liner filters rows containing zero, finds the first zero column, and slices three cells from the horizontally mirrored side of that same row. The references agree on a simple mirror-cutout rule; the only ambiguity is placement, where explicit fixtures show the ONNX solver must not assume the random branch top-left-only cutout range.
