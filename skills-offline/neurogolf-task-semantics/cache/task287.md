# task287 Semantics

## Sources

- Current champion builder: `solutions_py/task287.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task287.json`
- ARC-GEN task id: `b8825c91`
- ARC-DSL task id: `b8825c91`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_b8825c91.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_b8825c91.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task287.py`

## Pattern

The input is a 16x16 grid made from an 8x8 color bitmap mirrored into all four quadrants by main-diagonal and counter-diagonal symmetry. Color 4 is yellow and is used only as the background/cutout marker. The task cuts out two small rectangles by replacing their cells with yellow; the bundled known examples include cutouts outside the upper-left quadrant even though the random ARC-GEN branch samples upper-left coordinates. The output restores the complete symmetric pattern, so every yellow cutout cell is filled from its mirror-equivalent colored cell and no yellow remains.

## Readable Python Solver

```python
def solve(grid):
    n = len(grid)
    no_yellow = [[0 if v == 4 else v for v in row] for row in grid]

    main_filled = [
        [max(no_yellow[r][c], no_yellow[c][r]) for c in range(n)]
        for r in range(n)
    ]
    out = [
        [max(main_filled[r][c], main_filled[n - 1 - c][n - 1 - r]) for c in range(n)]
        for r in range(n)
    ]
    return out
```

## Generator Constraints

The base bitmap has size 8x8 and the task grid is always 16x16. For each upper-triangular coordinate in the 8x8 bitmap, the generator chooses a random non-yellow color and writes it symmetrically across the bitmap main diagonal. That 8x8 bitmap is then mirrored into all four 8x8 quadrants of the 16x16 grid. Exactly two cutout rectangles are applied only to the input, with width and height from 2 through 4. The ARC-GEN random branch chooses row/column starts in the upper-left 8x8 coordinate range, but the bundled train/test examples include explicit cutouts outside that range. The output is the uncut mirrored pattern. Cutouts may overlap mirror partners, so using the full main-diagonal and counter-diagonal symmetry relation is the safe fill rule.

## Reference Notes

ARC-DSL first replaces yellow with zero, mirrors across the main diagonal, and takes an elementwise maximum. It then mirrors that result across the counter-diagonal and takes another maximum. The final `cmirror` is redundant because the previous maximum already makes the grid counter-diagonal symmetric. The Code Golf solution expresses the same recurrence recursively: yellow cells are treated as false/zero, and otherwise values are pulled from reversed and transposed views.
