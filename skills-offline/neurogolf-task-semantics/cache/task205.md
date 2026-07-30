# task205 Semantics

## Sources

- Current champion builder: `solutions_py/task205.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task205.json`
- ARC-GEN task id: `8731374e`
- ARC-DSL task id: `8731374e`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_8731374e.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_8731374e.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task205.py`

## Pattern

The input is a 15x15 to 30x30 noisy grid containing one solid rectangular box. The box has width and height 6..10, is inset by at least one cell from the input border, and is filled with a box color. Inside the box are one to three marker cells of a second color, never on the box border. The output is exactly the cropped box. In that crop, every marker cell expands into a full horizontal and vertical cross of the marker color. Equivalently: copy the rectangular box crop, find the rare non-box color inside it, then fill every row and every column that contains that rare color with the rare color.

If multiple markers exist, their row/column crosses are unioned. The generator chooses distinct marker rows and distinct marker columns, zipped into marker coordinates, so there are no duplicate marker rows or columns in generated samples. The marker color and box color may be any two distinct digits.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda g, k=16: ~k * g or p([[min(r + c, key=K) for c in -k * g] or r for *r, in zip(*g) if ~-len({*r}) * 2 - k < 4 < r.count(max((l := sum(g, g)), key=(K := l.count)))], k - 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Input width and height are independently 15..30.
- The box width and height are independently 6..10.
- The box top-left offset is at least 1 from the input border and leaves at least one-cell margin on the bottom/right.
- The full input is random colors first; the box rectangle is then overwritten with one box color.
- A second color is sampled for markers. The two colors are distinct.
- There are 1..3 markers. Marker rows are sampled without replacement from `1..tall-2`; marker columns are sampled without replacement from `1..wide-2`; rows and columns are zipped, so marker coordinates have distinct rows and columns.
- Output size is the box size, not the original input size. NeuroGolf still requires the dense `[1,10,30,30]` tensor contract, with unused cells outside the output crop decoding to background.

## Reference Notes

ARC-GEN is direct: output starts as the box crop, then every row and column through each marker becomes the marker color. ARC-DSL finds the largest object, takes its subgrid, filters down to the regular box crop, chooses the least color inside that crop, then fills vertical and horizontal frontiers through those least-color cells. Code Golf 2025 expresses the same behavior compactly by iteratively filtering rows/columns and replacing cells according to the rare color/cross rule.
