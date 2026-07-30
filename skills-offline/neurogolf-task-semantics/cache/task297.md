# task297 Semantics

## Sources

- Current champion builder: `solutions_py/task297.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task297.json`
- ARC-GEN task id: `bd4472b8`
- ARC-DSL task id: `bd4472b8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_bd4472b8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_bd4472b8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task297.py`

## Pattern

The input has width `W` from 2 to 6 and height `2*W+2`. The first row contains `W` distinct non-gray colors. The second row contains gray (`5`) in those same active columns. All later rows are blank in the input.

The output preserves the first two rows, then appends two copies of the top row transposed into full constant-color rows: for each header color `colors[i]`, row `2+i` is filled with `colors[i]` across all active columns, and row `2+W+i` is filled with the same color across all active columns. Cells outside the active width/height remain background.

## Readable Python Solver

```python
def solve(grid):
    colors = [v for v in grid[0] if v != 0]
    w = len(colors)
    out = [row[:] for row in grid]
    for i, color in enumerate(colors):
        out[2 + i][:w] = [color] * w
        out[2 + w + i][:w] = [color] * w
    return out
```

## Generator Constraints

- Number of colors `W` is sampled from `2..6`.
- Colors are random non-gray colors; gray (`5`) is excluded from the top-row palette.
- Grid width is `W`; grid height is `2*W+2` before padding into the NeuroGolf 30x30 one-hot tensor.
- Row 0 is the palette/header row. Row 1 is all gray across the active width.
- For each color index `i`, output rows `2+i` and `2+W+i` are filled with `colors[i]` across the active width.
- All task examples have nonzero header colors, so color 0 is not an active header color.

## Reference Notes

The ARC-DSL solver crops the top two rows, mirrors the top row diagonally, horizontally upscales that mirrored row pattern by the grid width, repeats it twice, and vertically concatenates it after the header. The Code Golf solution `g[:2] + [*zip(*g[:1]*len(g[0]))] * 2` expresses the same transposed-row repeat.
