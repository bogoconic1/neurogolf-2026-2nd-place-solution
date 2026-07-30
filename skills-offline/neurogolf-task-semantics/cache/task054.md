# task054 Semantics

## Sources

- Current champion builder: `solutions_py/task054.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task054.json`
- ARC-GEN task id: `264363fd`
- ARC-DSL task id: `264363fd`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_264363fd.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_264363fd.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task054.py`

## Pattern

The input is a 30x30 scene with one or more large rectangular boxes/flags of a single `boxcolor` on a uniform `bgcolor`. Inside the boxes are one to four marker pixels using the guide star center color. Away from the boxes, usually near a corner after optional flip/transpose, there is a small guide star. The guide star encodes:

- `center_color`: the star center and marker color inside boxes.
- `line_color`: the color of the vertical and/or horizontal arms.
- `fill_color`: optional 3x3 fill/border color; it may be absent when the generator uses `-1`.
- `vert` and `horiz`: whether to draw vertical and/or horizontal lines through each marker. The guide has extra length-2 arms; target stars use only length-1 arms.

The output removes the guide star by restoring background there. It keeps the boxes and background, then for every marker inside a box it draws the requested horizontal line across that box row and/or vertical line down that box column in `line_color`. Finally it draws a target star centered on the marker: optional 3x3 fill, then center in `center_color`, then immediate vertical/horizontal arms in `line_color`. Lines are clipped to the containing box rectangle.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(r):
    f = [p * 1 for p in r]
    for p in range(28):
        for t in range(28):
            if r[p][t + 1] == r[p][t - 1] != r[p][t] != r[p + 1][t] == r[p - 1][t] != {r[p][t + 1], r[p + 1][t + 1]} != {r[p + 1][t]}:
                for e in range(28):
                    for n in range(28):
                        if r[p][t] == r[e][n]:
                            for o in range(-1, 2):
                                for h in range(-1, 2):
                                    i = 1
                                    if r[p + o * i][t + h * i] != r[p + 1][t + 2]:
                                        f[e + o * i][n + h * i] = r[p + o][t + h]
                                        i += 1
                                        if r[p][t] != r[p + o * i][t + h * i] != r[p + 1][t + 2]:
                                            while r[e + o * i][n + h * i] != r[p + 1][t + 2]:
                                                f[e + o * i][n + h * i] = r[p + o][t + h]
                                                i += 1
                for o in range(-2, 3):
                    for h in range(-2, 3):
                        f[p + o][t + h] = r[p + 1][t + 2]
    return f


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN normally uses size 30 and random two-box scenes, while validation examples include two and three boxes. Box widths are around 9-18 and heights around 7-17; boxes are separated and not near the guide. The guide is placed near rows 3-5 and cols 23-25 before optional vertical flip and transpose. There are one or two markers per box in random generation; validation examples include more. Marker rows/cols are strictly inside each box, with enough separation when two are sampled. `vert` and `horiz` are random but not both false. If both are true, the optional fill color can be absent (`-1`) or equal to the center color; otherwise a distinct fill/border color is present. Background, box, center, line, and fill colors are sampled as distinct colors before the fill may be overridden.

## Reference Notes

The ARC-DSL solver finds the smallest object as the guide, normalizes it, extracts guide height/width to infer vertical/horizontal line directions, removes the guide, finds occurrences of the guide center pattern in every remaining object, paints vertical and horizontal frontiers through each occurrence, shifts the normalized guide star onto each marker, then fills the guide footprint back with the scene background. The Code Golf solution scans for the guide star using local color-equality constraints, finds all pixels with the guide center color, copies guide-neighborhood colors around each marker, extends arms until the surrounding box/background boundary, and erases the guide footprint.
