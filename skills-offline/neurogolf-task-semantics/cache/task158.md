# task158 Semantics

## Sources

- Current champion builder: `solutions_py/task158.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task158.json`
- ARC-GEN task id: `6aa20dc0`
- ARC-DSL task id: `6aa20dc0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6aa20dc0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6aa20dc0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task158.py`

## Pattern

The input is a rectangular grid, at most `25x26` and embedded in the standard NeuroGolf `[1,10,30,30]` tensor. A small multicolor `3x3` sprite appears once fully visible. The same sprite also appears two to four total times as non-overlapping magnified copies with scale `1`, `2`, or `3`, possibly horizontally and/or vertically flipped. For copies after the first, only the two non-majority-color corner blocks are visible in the input; the repeated majority-color pixels are hidden as background. The output keeps the input grid size and background but fills every hidden copy with the full scaled/flipped sprite pattern.

The visible complete sprite is the object with the most distinct colors. Its majority color is the hidden fill color for the copies. The non-majority colored cells of that template are the anchor pattern to locate all transformed/upscaled copies.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(n):
    for h in range(12):
        d = 3 - h // 4
        n = [[*h] for h in zip(*n[::-1])]
        for e in range(len(n) + 1 - 3 * d):
            for g in range(len(n[0]) + 1 - 3 * d):
                i = [n[e + h // (3 * d)][g + h % (3 * d)] for h in range(3 * d * 3 * d)]
                if i[0] != i[8] != 3 < len({*i}):
                    p = i
    for h in range(12):
        d = 3 - h // 4
        n = [[*h] for h in zip(*n[::-1])]
        for e in range(len(n) + 1 - 3 * d):
            for g in range(len(n[0]) + 1 - 3 * d):
                i = [n[e + h // (3 * d)][g + h % (3 * d)] for h in range(3 * d * 3 * d)]
                if i[:d] == i[d * 3 * d - 3 * d:][:d] == p[:1] * d != i[-d:] == p[8:] * d != i[~d] == i[d]:
                    for h in range(3 * d * 3 * d):
                        n[e + h // (3 * d)][g + h % (3 * d)] = p[h // (3 * d) // d * 3 + h % (3 * d) // d]
    return n


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Width is `15..25`; height is width plus `-1`, `0`, or `1`.
- Four distinct colors are sampled: two unique corner colors, one repeated sprite color, and one background color.
- The base sprite is `3x3`, diagonally symmetric, and always contains colored cells at `(0,0)` and `(2,2)`. It samples two or three cells from `(1,0)`, `(1,1)`, `(2,0)`, `(2,1)` and mirrors non-diagonal choices across the diagonal. Those sampled/mirrored cells use the repeated hidden color.
- There are `2..4` placed sprites. The first placed sprite has magnification `1` and is fully visible in the input. Later sprites have magnification `1..3`.
- Placed sprites are non-overlapping with a margin of at least two cells.
- Each placed sprite may be horizontally flipped, vertically flipped, both, or neither.
- The output paints every full scaled/flipped sprite. The input paints every cell only for the first sprite; for later sprites it hides all cells except the non-majority diagonal corner colors.

## Reference Notes

- ARC-DSL selects the object with the most colors as the complete template, normalizes it, removes the template's most common color to get the visible anchor pattern, then searches occurrences of that anchor after mirror/scale transforms and paints shifted full templates back onto the input.
- The DSL transform set includes identity and mirror/diagonal variants; the generator itself uses horizontal and vertical flips. The sprite is diagonally symmetric enough that the broader transform search is safe on the task data.
- The Code Golf solution scans possible `3*d` square regions for the complete template, then rescans all scales/orientations for regions whose visible corner-color constraints match and fills them with the stored template.
