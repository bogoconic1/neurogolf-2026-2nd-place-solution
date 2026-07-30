# task213 Semantics

## Sources

- Current champion builder: `solutions_py/task213.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task213.json`
- ARC-GEN task id: `8e1813be`
- ARC-DSL task id: `8e1813be`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8e1813be.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8e1813be.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task213.py`

## Pattern

The input is a rectangular grid containing evenly spaced colored stripes, partly occluded by a gray square with a black border. The stripe colors are nonzero colors other than gray `5`; black `0` is background, border, and erasure.

The stripes are either horizontal or vertical. In the untransposed generator case, colored horizontal rows occur every three rows and each stripe has one solid color. The gray square plus black border hides a contiguous rectangular portion of the stripes. In the transposed case, the same construction is transposed, producing vertical stripes.

The output is a square whose side length is the number of stripe colors. If the input stripes are horizontal, each output row is filled with that stripe color in top-to-bottom order. If the input stripes are vertical, each output column is filled with that stripe color in left-to-right order. Equivalently, the output is the horizontal-stripe square transposed when the input was transposed.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: [l[o:] for r in g if (l := [e for e in r if e % 5])[~(o := (6 - len({*'%s' % g}))):]][o:]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Before optional transpose, height is `9..21` and width is `height..height+3`.
- Stripe offset is `0..2`; stripe rows are `range(offset, height, 3)`.
- The number of stripes is `(height + 2 - offset) // 3`.
- Stripe colors are sampled without replacement from `[1, 2, 3, 4, 6, 7, 8, 9]`, so gray `5` is never a stripe color.
- A gray square of side `num_stripes` is drawn with a one-cell black border, overwriting part of the stripes.
- The occluding square is placed touching the top-left corner, shifted down only, or shifted right only. This guarantees visible stripe evidence remains outside the square.
- The entire input and output may be transposed.
- Output side length is `num_stripes`, typically `3..7` under the generator.

## Reference Notes

The ARC-DSL solver replaces gray with black, detects whether the colored stripe object is vertical or horizontal, canonicalizes by diagonal mirror when needed, orders colored stripe objects by their uppermost coordinate, deduplicates their colors, repeats each color to square width, and mirrors the result back if the input was transposed.

The Code Golf 2025 solution is a compact row scan: for each row containing nonzero non-gray colors, it drops the offset introduced by the gray square and collects the stripe colors into the square output. Its behavior matches the generator because stripe colors are solid and gray is not a stripe color.
