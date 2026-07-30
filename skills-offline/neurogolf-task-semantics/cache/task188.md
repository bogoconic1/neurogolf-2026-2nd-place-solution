# task188 Semantics

## Sources

- Current champion builder: `solutions_py/task188.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task188.json`
- ARC-GEN task id: `7b7f7511`
- ARC-DSL task id: `7b7f7511`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7b7f7511.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7b7f7511.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task188.py`

## Pattern

The input is exactly two identical copies of a small colored tile. If the copy is stacked vertically, the input has shape `(2*height) x width` and the output is the top half. If the copy is side-by-side horizontally, the input has shape `height x (2*width)` and the output is the left half. The tile itself has width and height in `2..4` and uses 3 or 4 randomly chosen colors, with repetitions allowed. The task is therefore duplicate-removal: return one copy of the repeated half.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g: (X := g[:53 % ~-len(g)]) * (g == X + X) or [*map(p, g)]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Base tile width and height are each sampled from `2..4`.
- A color list of 3 or 4 colors is sampled, then every base-tile cell chooses from that list.
- `vert=1` stacks two identical copies vertically, so input dimensions are `width x (2*height)` in ARC row/column terms and output is the top copy.
- `vert=0` places two identical copies horizontally, so input dimensions are `(2*width) x height` and output is the left copy.
- Some vertical cases may be square at the padded-tensor level, so robust solvers should use equality of halves rather than only aspect ratio when possible.
- Output dimensions are the base tile dimensions, at most `4x4`.

## Reference Notes

The ARC-DSL solver branches on orientation and applies `tophalf` or `lefthalf`. The Code Golf 2025 solution recursively tries to split the outer list into two equal halves; if the whole grid is not a duplicate of its top half, it applies the same logic to each row, which handles horizontal duplication. This confirms the equality-based duplicate-removal interpretation.
