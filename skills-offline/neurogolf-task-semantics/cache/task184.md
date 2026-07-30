# task184 Semantics

## Sources

- Current champion builder: `solutions_py/task184.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task184.json`
- ARC-GEN task id: `780d0b14`
- ARC-DSL task id: `780d0b14`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_780d0b14.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_780d0b14.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task184.py`

## Pattern

The input is a noisy rectangular mosaic made from a small `tall x wide` grid of colored rooms, where `tall` and `wide` are each 2 or 3. Each room is a solid-color rectangle of random height/width 3..15, with about 10% of its cells erased to background `0`. Rooms are separated by full zero rows and full zero columns. The output is the compact `tall x wide` table of room colors, placed as the ARC output grid.

The task is to recover the room color for each separated row-band/column-band. Because each room has many cells and at most sparse erasures, the color can be recovered by the positive support/count inside the detected band intersection; zeros are separators or noise and should not become output colors.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    row_sep = [all(grid[r][c] == 0 for c in range(w)) for r in range(h)]
    col_sep = [all(grid[r][c] == 0 for r in range(h)) for c in range(w)]

    row_bands, start = [], None
    for r, sep in enumerate(row_sep + [True]):
        if not sep and start is None:
            start = r
        if sep and start is not None:
            row_bands.append(range(start, r))
            start = None

    col_bands, start = [], None
    for c, sep in enumerate(col_sep + [True]):
        if not sep and start is None:
            start = c
        if sep and start is not None:
            col_bands.append(range(start, c))
            start = None

    out = []
    for rows in row_bands:
        out_row = []
        for cols in col_bands:
            counts = [0] * 10
            for r in rows:
                for c in cols:
                    counts[grid[r][c]] += 1
            counts[0] = 0
            out_row.append(max(range(10), key=lambda color: counts[color]))
        out.append(out_row)
    return out
```

## Generator Constraints

- Input height/width are derived from 2 or 3 row bands and 2 or 3 column bands.
- Each room height and width is 3..15.
- A zero separator row/column of width 1 is inserted between adjacent bands.
- Each room has a randomly chosen nonzero color; colors may repeat across rooms.
- Each room cell is erased to zero with probability 1/10, but the room is large enough that the intended color remains the dominant nonzero support.
- Output size is exactly `tall x wide`, so 2x2, 2x3, 3x2, or 3x3.

## Reference Notes

The ARC-DSL solver finds connected/nonzero objects larger than size 2, paints their colors at their centers, clears the full canvas, then filters rows and columns that contain more than one distinct value after deduplication. This is equivalent to finding separated room bands and writing one representative color per band intersection.

The Code Golf solution recursively transposes/filters rows and uses maxima to collapse separated bands into the compact color table. It confirms that the essential operation is compressing nonzero room bands separated by all-zero rows/columns.
