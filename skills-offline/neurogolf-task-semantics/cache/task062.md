# task062 Semantics

## Sources

- Current champion builder: `solutions_py/task062.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task062.json`
- ARC-GEN task id: `2bcee788`
- ARC-DSL task id: `2bcee788`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_2bcee788.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_2bcee788.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task062.py`

## Pattern

The input is a 10x10 grid with black background (`0`), a small connected colored sprite, and red (`2`) marker cells on the opposite side. The colored sprite is generated inside a 3x3 bounding box and may be horizontally flipped and/or transposed. The red cells mark the visible edge/axis of the missing reflected copy. The output changes the background to green (`3`), keeps the original colored sprite, removes the red markers, and paints the missing horizontally or vertically reflected copy in the sprite color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: ((g := [(q := [i or 3 for i in r]) for *r, in zip(*(g[670:] or p(g * 2))) if (g := (g != ({*r} == {2, 3} < {2, *q})))])[:1] * 9 + g + g[::-1])[~9:]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

The generator fixes `size=10` and `minisize=3`. The sprite is a connected subset of a 3x3 bitmap grown from the left column and guaranteed to include at least one hidden (`c>0`) cell. The sprite anchor is sampled with `row in 1..6` and `col in 4..6`, then optional horizontal flip and transpose are applied. The color excludes red (`2`) and green (`3`). Red markers are placed at mirrored positions for sprite cells in source column `0`; hidden mirrored cells are black in the input and colored in the output.

## Reference Notes

The ARC-DSL solver replaces the most common color with green, finds the largest non-background object as the colored sprite and the smallest object as the red marker object, determines whether the marker is horizontal, mirrors the sprite horizontally or vertically, shifts the mirrored object next to the marker, and paints it. The Code Golf solution recursively/compactly performs the same background-to-green and reflected-sprite fill.
