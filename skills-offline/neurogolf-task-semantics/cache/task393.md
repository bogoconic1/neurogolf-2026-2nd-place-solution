# task393 Semantics

## Sources

- Current champion builder: `solutions_py/task393.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task393.json`
- ARC-GEN task id: `f8ff0b80`
- ARC-DSL task id: `f8ff0b80`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f8ff0b80.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f8ff0b80.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task393.py`

## Pattern

The input is a `12x12` grid containing three separated non-background colored connected objects. The objects have distinct sizes. Output is a `3x1` column containing the three object colors ordered from largest object area to smallest object area.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter

    counts = Counter(v for row in grid for v in row if v != 0)
    colors = [c for c, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)]
    return [[c] for c in colors[:3]]
```

## Generator Constraints

ARC-GEN uses fixed `size=12` and `num_boxes=3`. It samples three distinct object sizes from `3..14`, sorted descending. Each size determines a creature bounding side length: `3` for counts up to `6`, `4` for counts `7..12`, and `5` for counts above `12`. The generated creatures are continuous shapes placed without overlap and with a one-cell separation margin. Three random non-black colors are assigned to the objects. Because counts are sorted descending and sampled without replacement, the required output order is unique.

## Reference Notes

ARC-DSL finds connected objects, orders them by size, extracts their colors, converts each color to a `1x1` canvas, merges the canvases, and mirrors to produce the vertical output column. The Code Golf 2025 solution counts flattened color occurrences and takes the top three non-background colors. The references agree on area-ranked color output.
