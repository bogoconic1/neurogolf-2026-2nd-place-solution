# task189 Semantics

## Sources

- Current champion builder: `solutions_py/task189.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task189.json`
- ARC-GEN task id: `7c008303`
- ARC-DSL task id: `7c008303`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7c008303.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7c008303.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task189.py`

## Pattern

The input is a 9x9 grid containing a cyan (`8`) separator row and column, a
2x2 palette of non-cyan/non-green colors in one corner, and a 6x6 pattern area
containing green (`3`) marker cells. The output is the 6x6 pattern area with
each green marker recolored by the corresponding quadrant color from the 2x2
palette. The palette cell colors are expanded to four 3x3 quadrants; cells in
the 6x6 pattern area that were not green remain black (`0`). The whole input
and output may be flipped horizontally, vertically, or both, so the apparent
palette corner and cyan separator position vary, but the rule is the same after
orientation.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, h=[]: g * 0 != 0 and [*map(p, 3 * [g[(i := (('0' in '%r' % g[2]) * 7))]] + 3 * [g[i + 1]], (h + g)[3 >> i:])] or h % 2 * g


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN uses a fixed `size=6`, so inputs are always 9x9 and outputs are always
6x6. Before optional flips, row 2 and column 2 are cyan separators, the 2x2
palette is at rows/cols `0..1`, and the 6x6 marker area begins at `(3, 3)`.
The generator places a random set of green marker pixels in that 6x6 area.
Each marker receives the palette color selected by its output quadrant:
top-left, top-right, bottom-left, or bottom-right, with quadrant boundaries at
row/column 3. The full task may be flipped horizontally and/or vertically.
Palette colors are chosen from colors other than green and cyan and may repeat.
There is no tie-breaking ambiguity; the cyan separators and green marker
bounding box identify the oriented palette and output mask.

## Reference Notes

The ARC-DSL solver finds the green cells, takes their subgrid as the output mask, removes green and cyan from the input, compresses away empty rows/columns to recover the oriented 2x2 palette, upscales that palette by 3, and fills black into the zero positions of the green mask. The Code Golf solution uses the same idea tersely: choose the correct side of the cyan separator, repeat the palette rows/columns by three, and mask by the green pattern. The references agree with ARC-GEN.
