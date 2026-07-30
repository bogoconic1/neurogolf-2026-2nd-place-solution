# task313 Semantics

## Sources

- Current champion builder: `solutions_py/task313.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task313.json`
- ARC-GEN task id: `caa06a1f`
- ARC-DSL task id: `caa06a1f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_caa06a1f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_caa06a1f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task313.py`

## Pattern

The input is an `N x N` square, `5 <= N <= 20`, embedded in the standard
`30 x 30` NeuroGolf canvas. The visible task grid has a colored periodic
pattern in the upper-left portion and a uniform background-colored border along
the bottom and right. The output keeps the same `N x N` size and fills every
cell with the periodic color sequence implied by the non-background prefix:
`output[r][c] = colors[(r % 2 + c + 1) % len(colors)]`.

The palette has either two colors for sizes below 12 or three colors for larger
sizes. The background color is excluded from the palette. The border is only
evidence for the grid extent and background; it is replaced by the continued
periodic pattern in the output.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    bg = grid[-1][-1]

    # Recover the non-background colors from the first row prefix. The
    # generator emits a contiguous patterned prefix before the right border.
    colors = []
    for x in grid[0]:
        if x == bg:
            break
        if x not in colors:
            colors.append(x)
        if len(colors) == 3:
            break

    # Small generated grids use two colors; larger grids may use three. If the
    # first row alternates only two colors, scan the visible pattern for the
    # missing third color.
    if len(colors) < 3:
        for row in grid:
            for x in row:
                if x != bg and x not in colors:
                    colors.append(x)

    k = len(colors)
    return [[colors[(r % 2 + c + 1) % k] for c in range(w)] for r in range(h)]
```

## Generator Constraints

- The generated task grid is square with size `5..20`.
- `border` is `1..size//2`; all cells with `max(r, c) + border >= size` are
  background in the input.
- The background color `b` is random and is not in the periodic palette.
- The palette length is `2 + size // 12`, so it is 2 colors for sizes `5..11`
  and 3 colors for sizes `12..20`.
- For the non-border input prefix, `grid[r][c] = colors[(r % 2 + c) % k]`.
- For every output cell, including the former border, `output[r][c] =
  colors[(r % 2 + c + 1) % k]`.
- There is no object ambiguity in the generator: one periodic square plus a
  bottom/right background border.

## Reference Notes

ARC-GEN gives the exact closed-form rule and the only branch: two-color vs three-color palettes by grid size. ARC-DSL solves the same task more generally by painting the input into a larger background canvas, finding the periodic object, measuring vertical and horizontal periods, and tiling shifted copies back over the original grid. The Code Golf solution is a recursive periodic fill that relies on the first rows and generated row/column periodicity.

No disagreement was found between the references. For NeuroGolf, the standard input/output tensors remain `[1, 10, 30, 30]`; only the active task square inside the canvas has semantic content.
