# task281 Semantics

## Sources

- Current champion builder: `solutions_py/task281.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task281.json`
- ARC-GEN task id: `b548a754`
- ARC-DSL task id: `b548a754`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b548a754.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b548a754.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task281.py`

## Pattern

The input is an 11x11 to 13x13 grid with black background, one colored rectangular box near the top/left, and one cyan marker (`8`) below the box. The box has an outer border color and a different inner fill color. The marker column lies within the box width. The output extends the box vertically from its original top row down through the marker row, preserving the original width and border/fill pattern: outer color on the rectangle border, inner color in the interior, and black outside the extended rectangle. The cyan marker disappears. The whole example may be transposed and/or vertically flipped by the generator, so the solver must recover the occupied bounds and marker-aware extension after those transforms.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: exec('*_,a,b,c=map(g.index,filter(any,g));g[:]=zip(*(g,g[:a]+(c-a)*[g[a]]+[g[b]]+g[c+1:])[b<a<c][::-1]);' * 4) or g


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

Raw width and height are independently `11..13`. The original box starts at row/col `0..3`, with width and height `3..5`. The cyan marker is below the box before optional transforms: `dotrow = row + tall + gap`, where the gap is at least 1 and keeps the marker inside the grid; `dotcol` is within the box columns. The two box colors are random non-cyan colors. The generated pair may be transposed, then vertically flipped. These transforms mean the apparent extension direction varies, but the final output is always the bounding box of all nonzero cells filled as an outer/inner rectangle after removing cyan from the palette.

## Reference Notes

ARC-DSL finds objects including the marker, replaces cyan with zero for color analysis, picks the least frequent remaining colors as the inner/outer box colors, merges nonzero objects to get the extended bounding box/backdrop, then fills the backdrop with one color and the border box with the other. The Code Golf solution repeatedly transposes/flips until the rows are in the convenient orientation, then repeats box rows to span down to the marker. Both references agree that the output is a filled outer/inner rectangle over the combined box+marker bounds.
