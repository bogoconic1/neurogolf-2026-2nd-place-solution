# task307 Semantics

## Sources

- Current champion builder: `solutions_py/task307.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task307.json`
- ARC-GEN task id: `c59eb873`
- ARC-DSL task id: `c59eb873`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c59eb873.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c59eb873.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task307.py`

## Pattern

The input is a square grid of side `2..5` with black background and a subset of colored cells. The output is the input upscaled by factor 2 in both spatial dimensions: every input cell becomes a solid `2x2` block of the same color. Black cells become black `2x2` blocks and colored cells become same-color `2x2` blocks.

In NeuroGolf tensor form, the model receives the standard padded one-hot `[1,10,30,30]` input. Only the top-left `size x size` region is meaningful. The output is the upscaled grid in the top-left `2*size x 2*size` region with zeros outside that logical output area.

## Readable Python Solver

```python
def solve(grid):
    out = []
    for row in grid:
        expanded = []
        for value in row:
            expanded.extend([value, value])
        out.append(expanded[:])
        out.append(expanded[:])
    return out
```

## Generator Constraints

- Input side length is randomly chosen from `2..5`.
- Foreground pixels are a nonempty random subset of the square grid.
- The number of available foreground colors is `size + 1`, drawn from ordered palette `(1, 2, 3, 5, 6, 7, 8, 9, 4)`; black is the background.
- The generator calls `common.grid_enhance(size, 2, ...)`, which performs the factor-2 enhancement and keeps black as background.
- Hand-authored examples cover sizes `2`, `3`, `4`, and `5`.

## Reference Notes

The ARC-DSL solver is exactly `upscale(I, TWO)`. The Code Golf 2025 solution recursively duplicates each row and each element, another direct expression of 2x nearest-neighbor upscaling. There is no color-dependent logic, object detection, or tie-breaking.
