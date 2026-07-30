# task014 Semantics

## Sources

- Current champion builder: `solutions_py/task014.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task014.json`
- ARC-GEN task id: `0b148d64`
- ARC-DSL task id: `0b148d64`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_0b148d64.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_0b148d64.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task014.py`

## Pattern

The input is a 15..25 by 15..25 grid with two nonzero colors arranged in the four corner quadrants around a blank horizontal/vertical separator band.  One randomly chosen quadrant uses the rare color and the other three quadrants use the common color; pixels inside each quadrant are noisy/sparse and separator bands remain zero.  The required output is the tight rectangular crop around all cells of the rarest nonzero color, preserving the rare-color pixels and zero holes inside that bounding box.  The output size is the rare quadrant bounding box, not the whole quadrant and not recolored.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter

    h, w = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row if v != 0)
    rare = min(counts, key=counts.get)

    rows = [r for r in range(h) for c in range(w) if grid[r][c] == rare]
    cols = [c for r in range(h) for c in range(w) if grid[r][c] == rare]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return [row[c0:c1 + 1] for row in grid[r0:r1 + 1]]
```

## Generator Constraints

- Full grid width and height are each 15..25.
- A horizontal separator band has thickness 2..5 and starts at a row chosen from `5..height-5-rowthick`.
- A vertical separator band has thickness 2..5 and starts at a column chosen from `5..width-5-colthick`.
- Four corner quadrants exist around the separator cross.  Each non-separator quadrant is populated at about 90% density with one of two nonzero colors; separator cells and random holes are zero.
- Exactly one quadrant uses the rare color.  The generator retries until that rare quadrant color is strictly the rarest foreground color.
- The rare-color bounding box can be as small as roughly 5x5 and as large as about 14x14, depending on quadrant size and random holes.  The output includes zero holes inside the rare-color bbox.

## Reference Notes

- ARC-DSL solves this as `partition(I) -> argmin(size) -> subgrid`, i.e. find the smallest foreground object/component and crop its bounding box from the original grid.
- The ARC-GEN implementation computes the rarest nonzero color, then directly takes min/max row and column coordinates for that color.
- The Code Golf solution is a terse recursive row/column filter for rows/columns that contain the mixed rare-color structure; it agrees with the tight crop behavior.
