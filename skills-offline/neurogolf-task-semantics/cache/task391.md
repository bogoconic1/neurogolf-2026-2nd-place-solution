# task391 Semantics

## Sources

- Current champion builder: `solutions_py/task391.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task391.json`
- ARC-GEN task id: `f8b3ba0a`
- ARC-DSL task id: `f8b3ba0a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f8b3ba0a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f8b3ba0a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task391.py`

## Pattern

The input is a sparse framed encoding of a small `height x width` bitmap. The encoded bitmap has width `3*w + 1` and height `2*h + 1`; only cells at odd rows and the first two columns of each 3-column block carry bitmap colors. The remaining frame/separator cells are black `0`. The bitmap itself uses four nonzero colors: one majority/base color and three minority colors. The output is a `3x1` column containing the three minority colors ordered by descending frequency in the encoded bitmap.

## Readable Python Solver

```python
from collections import Counter

def solve(grid):
    counts = Counter(v for row in grid for v in row if v != 0)
    base = max(counts, key=counts.get)
    ranked = sorted(
        ((n, c) for c, n in counts.items() if c != base),
        reverse=True,
    )
    return [[c] for n, c in ranked]
```

## Generator Constraints

ARC-GEN chooses bitmap width `3..5` and height `6..7`. It draws each logical bitmap cell as a two-cell horizontal bar inside a zero separator lattice, so color counts in the input are exactly twice the logical bitmap counts. Four random colors are used in the bitmap. The base color fills all logical cells first; three other colors are then placed at distinct counts sampled from `1..4`. The base color count is therefore always the largest, and the three output colors have distinct counts, so there is no tie-breaking ambiguity in generated cases.

## Reference Notes

ARC-DSL compresses away the zero lattice, counts the palette colors in the compressed bitmap, orders colors by decreasing count via inverted `colorcount`, makes one-cell canvases for each ordered color, merges them, and crops away the first row. Cropping removes the majority/base color and leaves the three minority colors in descending count order. The Code Golf 2025 solution performs the same idea by sorting color digits by their flattened grid count and taking the three colors just below the two dominant colors (`0` lattice and bitmap base color).
