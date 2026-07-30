# task261 Semantics

## Sources

- Current champion builder: `solutions_py/task261.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task261.json`
- ARC-GEN task id: `a79310a0`
- ARC-DSL task id: `a79310a0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_a79310a0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_a79310a0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task261.py`

## Pattern

The input is a square black grid containing one nonempty cyan (`8`) object. The output is the same size as the input, but the cyan object is shifted down by exactly one row and recolored red (`2`). All other cells are black (`0`). The original cyan cells become black because only the shifted copy appears in the output.

In NeuroGolf one-hot tensor terms, output channel `2` is input channel `8` shifted down by one row, and output channel `0` is the black background everywhere except those shifted red cells. There are no ties or multiple-object choices: all cyan pixels are treated as one object and shifted together.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            if color == 8:
                out[r + 1][c] = 2
    return out
```

## Generator Constraints

ARC-GEN chooses a square size from `3..7`. It samples a rectangular window with width and height around half the grid size, then places a random nonempty subset of pixels inside that window. The object is always cyan (`8`) on a black (`0`) background. The top-left row for the sampled window is chosen so the object has at least one blank row below it after shifting: `row` is selected from `0..size-height-1`, so every cyan pixel satisfies `r + 1 < size`. The object may be disconnected, may have repeated columns or rows, and may contain one pixel. The output is freshly black and only writes red (`2`) at shifted cyan positions.

Validation examples include a 2x2 block at the top-left, a single cyan pixel in a 3x3 grid, a horizontal run, and a mixed shape. These cover singletons, multi-pixel components, horizontal and vertical adjacency, and shapes that are not full rectangles.

## Reference Notes

The ARC-DSL solver gets the cyan object with `objects(I, T, F, T)`, takes the first object, moves it `DOWN`, then replaces `EIGHT` with `TWO`. That confirms the operation is object-level translation plus recolor, not line drawing or fill.

The Code Golf 2025 solution prepends the last row and drops it from the bottom, then maps each cell through `x % 6`, which sends `8 -> 2` and leaves `0 -> 0`. This works because the generator guarantees the final row is black and no cyan pixel would shift out of bounds.
