# task192 Semantics

## Sources

- Current champion builder: `solutions_py/task192.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task192.json`
- ARC-GEN task id: `7e0986d6`
- ARC-DSL task id: `7e0986d6`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7e0986d6.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7e0986d6.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task192.py`

## Pattern

The input contains a black background, several filled rectangles of one
foreground color, and isolated single-pixel noise of a second foreground color.
The output has the same height and width. All noise pixels outside rectangles
are removed to black. Noise pixels that overwrote cells inside a rectangle are
repaired back to the rectangle color, so the final image is exactly the union of
the filled rectangles.

There are only two nonzero colors in normal generated cases: the sparse noise
color and the rectangle color. The noise color is the least frequent nonzero
color; the rectangle color is the other nonzero color. Rectangles are separated,
axis-aligned, and same-colored. Because noise is isolated, deleting the noise
color and filling the bounding box of each remaining rectangle component
recovers the intended filled boxes.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
Z = (0,)
p = lambda g: [*map((F := (lambda *r: [max(x, a | b, key=(Z * 99 + r).count) for a, b, x in zip(Z + r, r[1:] + Z, r)])), *map(F, *g))]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Dimensions are random width and height in `10..20`.
- There are `3..5` filled rectangles. Each rectangle has width and height in
  `3..10`.
- Rectangles have the same `boxcolor`, are axis-aligned, and are non-overlapping
  with `spacing=1`, so a one-cell gap between boxes is allowed.
- The noise color is a different random color. Noise pixels are sampled over the
  whole grid with probability `0.05`, then filtered by `remove_neighbors`, so
  noise pixels are isolated from each other but can lie outside, adjacent to, or
  inside rectangles.
- The generator writes rectangles first, then writes noise pixels into the
  input only. The target output keeps the original rectangles and contains no
  noise-color cells.
- Fresh edge case: a noise pixel in the one-cell gap between two boxes can have
  two direct rectangle-color neighbors but should still be removed. This is why
  a purely local "restore if two neighbors are box color" rule is unsafe.

## Reference Notes

- ARC-DSL identifies the rare color, replaces it with black, identifies the box color, then restores rare-color cells whose direct neighbors contain more than one box-color cell. This repairs holes inside rectangles on the canonical examples.
- The Code Golf 2025 solution performs a compact two-pass local majority cleanup over rows and columns. It expresses the same local intuition: isolated noise should be overwritten by the surrounding dominant color or black.
- These reference solutions are useful for understanding the visual rule, but they expose an unsafe shortcut for the broader generator. Local neighbor repair can accidentally fill bridge/gap noise between adjacent rectangles. The generator-exact interpretation is rectangle recovery, not arbitrary two-neighbor filling.
