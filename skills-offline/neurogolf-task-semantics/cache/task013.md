# task013 Semantics

## Sources

- Current champion builder: `solutions_py/task013.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task013.json`
- ARC-GEN task id: `0a938d79`
- ARC-DSL task id: `0a938d79`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_0a938d79.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_0a938d79.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task013.py`

## Pattern

The input rectangle contains exactly two nonzero marker pixels in different colors. In landscape orientation, the markers sit on the top or bottom border at columns `start` and `start + sep + 1`; the output fills vertical stripes from `start` onward, alternating the two marker colors every `sep + 1` columns. In portrait orientation the task is transposed, so marker rows define horizontal stripes. Padded area outside the original rectangle remains background.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, h=0, l=[]: [[(l := ([max(c + [j * (i > 0) for i, j in zip(l, l[1::2])])] + l))[:1] * len(c), c][c[1:-1] > c] for *c, in zip(*(h or p(g, g)))]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Valid rectangle size before padding: width `20..30`, height `6..12`.
- `start` is in `1..width//2`; `sep` is `1..5`; exactly two colors are used.
- The second marker is at `start + sep + 1`, so the stripe period is the marker distance.
- Markers may be on the top or bottom border independently.
- `xpose` optionally transposes both input and output. The NeuroGolf tensor remains padded to 30x30.
- The valid rectangle is complete one-hot input including background cells; padded cells outside it are all-zero across channels.

## Reference Notes

- ARC-DSL transposes portrait inputs to a landscape framing, partitions the two marker objects, computes the stripe period from marker width/distance, paints frontiers at alternating offsets, then transposes back if needed.
- ARC-GEN confirms that because valid cells are a full rectangle of one-hot pixels, area and first row/column moments recover `height-1` and `width-1` exactly.
- Code Golf recursively works through transposed rows/columns and propagates alternating stripe labels.
