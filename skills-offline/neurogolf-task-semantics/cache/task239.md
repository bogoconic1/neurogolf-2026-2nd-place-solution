# task239 Semantics

## Sources

- Current champion builder: `solutions_py/task239.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task239.json`
- ARC-GEN task id: `9af7a82c`
- ARC-DSL task id: `9af7a82c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_9af7a82c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_9af7a82c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task239.py`

## Pattern

The input is a small 3x3, 3x4, 4x3, or 4x4 colored grid. Count the number of cells of each nonzero color. Sort colors by decreasing count. The output is a compact vertical bar chart: one column per present color, ordered by decreasing count, with each column filled from the top down by that color for exactly its count. The output height is the maximum count and the output width is the number of present colors; NeuroGolf pads this to the fixed 30x30 one-hot canvas.

## Readable Python Solver

```python
def solve(grid):
    cells = [x for row in grid for x in row if x]
    counts = {}
    for color in cells:
        counts[color] = counts.get(color, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    height = ordered[0][1]
    out = [[0 for _ in ordered] for _ in range(height)]
    for c, (color, count) in enumerate(ordered):
        for r in range(count):
            out[r][c] = color
    return out
```

## Generator Constraints

ARC-GEN chooses width and height in 3..4. It fills nested rectangular regions with colors from a random palette until every used color has a unique cell count; this guarantees a deterministic decreasing-count order without ties. Generated colors are nonzero.

## Reference Notes

ARC-DSL extracts color objects, orders them by size, builds one colored vertical strip per object with zero padding below to match the maximum count, mirrors/merges the strips into the output. The Code Golf 2025 solution counts flattened colors and sorts by negative count, then transposes padded columns. All references agree that the transformation is a histogram by color frequency, not a geometric copy.
