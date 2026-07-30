# task208 Semantics

## Sources

- Current champion builder: `solutions_py/task208.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task208.json`
- ARC-GEN task id: `890034e9`
- ARC-DSL task id: `890034e9`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_890034e9.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_890034e9.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task208.py`

## Pattern

The input is a `21 x 21` grid with two rectangular black cutouts of the same width and height, embedded in a noisy two-color background. One cutout already has a one-cell-thick rectangular frame in a rare box color; the other cutout is still unframed. The output keeps the grid unchanged except that it draws the same one-cell-thick frame, in the same box color, around the second black cutout. The rectangle size is inferred from the existing framed cutout; both cutouts have identical interior dimensions.

## Readable Python Solver

```python
def solve(grid):
    H, W = len(grid), len(grid[0])
    colors = sorted({v for row in grid for v in row})
    # The frame color is the least frequent non-background/noise color in ARC-DSL
    # terms (`leastcolor`). It is the color of the already drawn box border.
    counts = {c: sum(row.count(c) for row in grid) for c in colors}
    box = min(colors, key=lambda c: counts[c])

    border = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == box}
    r0, r1 = min(r for r, c in border), max(r for r, c in border)
    c0, c1 = min(c for r, c in border), max(c for r, c in border)
    h = r1 - r0 - 1
    w = c1 - c0 - 1

    out = [row[:] for row in grid]
    for row in range(1, H - h):
        for col in range(1, W - w):
            if all(grid[row + dr][col + dc] == 0 for dr in range(h) for dc in range(w)):
                for rr in range(row - 1, row + h + 1):
                    out[rr][col - 1] = box
                    out[rr][col + w] = box
                for cc in range(col - 1, col + w + 1):
                    out[row - 1][cc] = box
                    out[row + h][cc] = box
    return out
```

## Generator Constraints

ARC-GEN fixes the canvas size to `21`. The black cutout interior width and height are sampled from `2..5`, excluding the tiny `2 x 2` case. Two cutouts are placed with enough row or column separation that their boxes do not overlap. The background uses two random non-box colors plus rare black noise pixels, but the generator rejects accidental extra all-black rectangles of the target size. The box color is distinct from the background colors and is drawn around only the first cutout in the input; the output draws it around both.

## Reference Notes

The ARC-DSL solver takes `leastcolor(I)` as the frame color, extracts that color's object, computes its `inbox` (the framed black interior), recolors that interior pattern to zero, finds all `occurrences` of that zero rectangle in the input, shifts the normalized frame border to those occurrences, and fills them with the frame color. The Code Golf solution uses regex substitution to find the rare color and replicate the frame pattern around matching zero regions.
