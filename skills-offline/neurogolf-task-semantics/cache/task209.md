# task209 Semantics

## Sources

- Current champion builder: `solutions_py/task209.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task209.json`
- ARC-GEN task id: `8a004b2b`
- ARC-DSL task id: `8a004b2b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8a004b2b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8a004b2b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task209.py`

## Pattern

The input contains a large yellow (`4`) rectangular frame, a partially shown magnified sprite inside that frame, and the original small sprite near the bottom of the grid. The small sprite is a connected creature with width `w` and height `t`; its cells use colors from `{1,2,3,8}`. The magnified copy uses an integer magnification factor `mag` in `2..4`, but only a subset of the small sprite cells are shown inside the yellow frame. The output is the full framed rectangle cropped from the input, with the missing magnified sprite cells filled in from the complete small sprite.

The yellow frame corners remain yellow. All non-frame output cells match the interior of the framed input region, except that every small sprite cell is expanded to a `mag x mag` block at the corresponding position inside the frame.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(g):
    e = [(n ^ 4, r, o) for r, g in enumerate(g) for o, n in enumerate(g) if n]
    for o, n in enumerate(g):
        for q, j, n in e[e.index(sorted(e)[3]) + 1:]:
            d = [r * 9 for r in g * 2]
            f = 5
            for q, r, l in e[e.index(sorted(e)[3]) + 1:]:
                r = e[2][1] + o * (r - j)
                f += o * o * (d[r][e[2][2] + o * (l - n)] == q ^ 4) + 1
                exec(o * 'd[r][e[2][2]+o*(l-n):e[2][2]+o*(l-n)+o]=o*[q^4];r+=1;')
            if f > len(e):
                return [r[e[0][2]:sorted(e)[3][2] + 1] for r in d[e[0][1]:sorted(e)[3][1] + 1]]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Input width and height are each `15..20`.
- Magnification `mag` is `2..4` and bounded by the input height.
- The original sprite has width `w` in `2..(7-mag)` and height `t` in `2..3`; it has `max(4, w*t//2)..w*t` connected cells.
- Sprite colors are sampled independently from `{1,2,3,8}`.
- The yellow frame has four yellow corners only, with width at least `mag*w+2` and height at least `mag*t+2`.
- The magnified sprite is placed strictly inside the frame, and only a random subset of sprite cells is shown in the input.
- The complete original sprite is near the bottom: `srow = height - t - 1`.

## Reference Notes

The ARC-DSL solver crops the yellow frame, selects the lowermost object as the original sprite, removes yellow from the crop, finds the shown magnified object inside the crop, infers `mag` as magnified width divided by original width, upscales the normalized original sprite, shifts it to the shown magnified object's upper-left corner, and paints it into the crop.

The Code Golf solution searches candidate magnification factors and anchor positions against the shown cells, then fills the crop.
