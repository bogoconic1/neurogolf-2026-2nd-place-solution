# task235 Semantics

## Sources

- Current champion builder: `solutions_py/task235.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task235.json`
- ARC-GEN task id: `995c5fa3`
- ARC-DSL task id: `995c5fa3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_995c5fa3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_995c5fa3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task235.py`

## Pattern

The input is a fixed 4x14 grid containing three 4x4 gray glyph slots at columns `0..3`, `5..8`, and `10..13`, separated by one black column. Each slot encodes one output color from `{2, 3, 4, 8}` by the location of black cells inside the gray block:

- color `2`: no black holes; the 4x4 slot is all gray.
- color `8`: the centered 2x2 cells at rows `1..2`, cols `1..2` are black.
- color `3`: the side cells at rows `1..2`, cols `0` and `3` are black.
- color `4`: the lower centered 2x2 cells at rows `2..3`, cols `1..2` are black.

The output is a 3x3 grid. Output row `i` is filled entirely with the decoded color of slot `i`.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(3)] for _ in range(3)]
    for i, c0 in enumerate((0, 5, 10)):
        # These three probes distinguish the four generator glyphs.
        center_top_black = grid[1][c0 + 1] == 0
        side_black = grid[1][c0] == 0
        lower_center_black = grid[3][c0 + 1] == 0
        if center_top_black:
            color = 8
        elif side_black:
            color = 3
        elif lower_center_black:
            color = 4
        else:
            color = 2
        out[i] = [color, color, color]
    return out
```

## Generator Constraints

- Input shape is always 4 rows by 14 columns.
- The three slots are fixed 4x4 blocks at x offsets `0`, `5`, and `10`; there is no dynamic object search.
- Slots start as all gray (`5`), then black (`0`) holes are carved according to one of the four color templates.
- Each of the three slots independently chooses from color list `(2, 3, 4, 8)`. Repetition is allowed.
- Output shape is always 3x3. Each row is constant and depends only on the corresponding slot.

## Reference Notes

- ARC-DSL splits the input horizontally into three pieces, measures the black-cell object in each piece, maps its upper-left corner and size to one of four row colors, then horizontally upscales each row by three.
- The Code Golf solution uses a compact arithmetic expression over row 2/3 probes at slot offsets; it confirms that only a few fixed probes are needed and no full glyph matching is necessary.
- The no-hole color `2` is the default case when none of the black patterns is present.
