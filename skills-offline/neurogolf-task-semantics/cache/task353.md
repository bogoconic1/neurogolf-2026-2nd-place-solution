# task353 Semantics

## Sources

- Current champion builder: `solutions_py/task353.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task353.json`
- ARC-GEN task id: `dc433765`
- ARC-DSL task id: `dc433765`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_dc433765.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_dc433765.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task353.py`

## Pattern

The input has one green (`3`) pixel and one yellow (`4`) pixel on a black background. The yellow pixel lies in the same row, same column, or same diagonal as the green pixel, with at least one cell of separation. The output keeps the yellow pixel fixed and moves the green pixel exactly one step toward the yellow pixel along that row/column/diagonal direction. The old green position becomes black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    green = yellow = None
    for r, row in enumerate(grid):
        for c, v in enumerate(row):
            if v == 3:
                green = (r, c)
            elif v == 4:
                yellow = (r, c)
    gr, gc = green
    yr, yc = yellow
    dr = (yr > gr) - (yr < gr)
    dc = (yc > gc) - (yc < gc)
    out[gr][gc] = 0
    out[gr + dr][gc + dc] = 3
    out[yr][yc] = 4
    return out
```

## Generator Constraints

ARC-GEN chooses width `3..12` and height equal to width plus `0..2`. It chooses a green source pixel and then chooses the yellow pixel from positions that share a row, column, main diagonal, or anti-diagonal with the green pixel while excluding the surrounding `3x3` neighborhood, so the one-step move never collides with yellow. Generated grids contain exactly one green and one yellow pixel.

The NeuroGolf tensor is still padded to `[1,10,30,30]`, but the meaningful grid is at most `14x12` from the generator. Public validation includes tiny `3x3` cases, vertical/horizontal alignments, and both diagonal directions.

## Reference Notes

ARC-DSL extracts the green and yellow cells, subtracts coordinates, takes the sign of the row/column delta, recolors the green object, and moves it by that signed unit vector. This confirms all eight compass directions are possible.

The Code Golf 2025 solution rotates the grid repeatedly and uses a list/pop trick to move green toward yellow under different orientations, but it expresses the same one-step move rule.
