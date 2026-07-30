# task159 Semantics

## Sources

- Current champion builder: `solutions_py/task159.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task159.json`
- ARC-GEN task id: `6b9890af`
- ARC-DSL task id: `6b9890af`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6b9890af.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6b9890af.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task159.py`

## Pattern

The input is a 15x15 to 30x30 grid containing two separate objects: a small colored 3x3 sprite and a red square magnifier frame. The sprite color is any non-red color and its occupied cells form a diagonally connected Conway-style 3x3 pattern. The frame is a red border of size `3*m + 2`, where `m` is 1..4. The output is exactly the framed square crop: keep the one-cell red border and paint the sprite inside it, magnified by factor `m`, with each occupied sprite cell becoming an `m x m` block of the sprite color. Empty sprite cells remain black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    red = 2

    red_cells = [(r, c) for r in range(h) for c in range(w) if grid[r][c] == red]
    r0, r1 = min(r for r, _ in red_cells), max(r for r, _ in red_cells)
    c0, c1 = min(c for _, c in red_cells), max(c for _, c in red_cells)
    outsize = r1 - r0 + 1
    m = (outsize - 2) // 3

    sprite_cells = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] not in (0, red)]
    sr0 = min(r for r, _, _ in sprite_cells)
    sc0 = min(c for _, c, _ in sprite_cells)
    color = sprite_cells[0][2]

    out = [[0 for _ in range(outsize)] for _ in range(outsize)]
    for i in range(outsize):
        out[0][i] = out[outsize - 1][i] = red
        out[i][0] = out[i][outsize - 1] = red
    for r, c, _ in sprite_cells:
        rr = r - sr0
        cc = c - sc0
        for dr in range(m):
            for dc in range(m):
                out[1 + rr * m + dr][1 + cc * m + dc] = color
    return out
```

## Generator Constraints

- Input width and height are independently 15..30.
- The sprite is a non-red 3x3 pattern from `common.conway_sprite()` and is diagonally connected.
- Magnification `m` is 1..4. Output/frame size is `3*m + 2`.
- The red frame and sprite are placed so their bounding boxes do not overlap in both axes.
- The red frame is only a one-cell border in the input; its interior is black.
- Output size is 5, 8, 11, or 14 depending on `m`, but NeuroGolf output tensor remains `[1,10,30,30]`.

## Reference Notes

ARC-DSL finds all objects, chooses the smallest object as the sprite, takes the red subgrid, computes `m = width(red_box) / 3` after accounting for the frame, upscales the normalized sprite by `m`, shifts it by one cell, and paints it into the red frame crop. Code Golf 2025 encodes the same operation tersely by extracting the red frame and non-red sprite rows.
