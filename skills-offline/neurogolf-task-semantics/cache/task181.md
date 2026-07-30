# task181 Semantics

## Sources

- Current champion builder: `solutions_py/task181.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task181.json`
- ARC-GEN task id: `760b3cac`
- ARC-DSL task id: `760b3cac`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_760b3cac.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_760b3cac.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task181.py`

## Pattern

The input is a 6x9 grid containing a cyan sprite fragment in the central 3x3 block and a fixed yellow reference shape in the lower middle. The output preserves the input and adds a horizontally mirrored copy of the cyan 3x3 sprite to the empty side of the yellow reference. If the yellow reference is in its unflipped orientation, the mirror is placed in the left 3 columns; if the whole grid is horizontally flipped, the mirror is placed in the right 3 columns.

## Readable Python Solver

```python
def solve(grid):
    out = [row[:] for row in grid]
    # The center 3x3 cyan source is rows 0..2, cols 3..5.
    # Yellow at (3, 3) means unflipped; otherwise yellow at (3, 5) means flipped.
    place_left = grid[3][3] == 4
    dst0 = 0 if place_left else 6
    for r in range(3):
        for c in range(3):
            if grid[r][3 + c] == 8:
                out[r][dst0 + (2 - c)] = 8
    return out
```

## Generator Constraints

ARC-GEN defaults to width 9 and height 6. It draws a cyan Conway-style sprite subset inside a 3x3 coordinate frame, excluding one connected five-cell yellow-like shape. The cyan source is placed at columns 3..5 and rows 0..2. The fixed yellow reference occupies cells `(3,3),(4,3),(4,4),(4,5),(5,4)`. With `flip=1`, both input and output are horizontally flipped across the 9-column grid. Output contains the original cyan source plus a mirrored cyan copy at columns 0..2 for unflipped cases or 6..8 for flipped cases. Colors are cyan 8, yellow 4, and background 0.

## Reference Notes

ARC-DSL finds the cyan cells, uses the upper-left yellow cell to determine orientation, vertically mirrors the cyan object in DSL coordinates (horizontal mirror in grid terms), shifts it by three columns left or right, and fills cyan. The Code Golf solution checks `g[3][3]` to choose side and writes `l[x:x+3] = l[5:2:-1]` for the top three rows, confirming the operation is a mirrored copy of the central 3-column cyan block. No reference conflict was found.
