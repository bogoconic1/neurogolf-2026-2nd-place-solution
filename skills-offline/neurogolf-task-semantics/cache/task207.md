# task207 Semantics

## Sources

- Current champion builder: `solutions_py/task207.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task207.json`
- ARC-GEN task id: `88a62173`
- ARC-DSL task id: `88a62173`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_88a62173.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_88a62173.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task207.py`

## Pattern

The input is a fixed `5 x 5` grid containing four separated `2 x 2` sprites at the corners: top-left, top-right, bottom-left, and bottom-right. Each sprite uses the same single nonzero color on a subset of its four cells and black elsewhere. Three sprites have the same subset pattern; one sprite has a different subset pattern. The output is the `2 x 2` sprite pattern belonging to the different corner, preserving the same nonzero color and black cells.

The ARC-DSL solver splits the grid into the four corner `2 x 2` halves and returns the least common of those four blocks. There is no tie in generated tasks: exactly three blocks are identical and one is different.

## Readable Python Solver

```python
def solve(grid):
    blocks = [
        [row[0:2] for row in grid[0:2]],
        [row[3:5] for row in grid[0:2]],
        [row[0:2] for row in grid[3:5]],
        [row[3:5] for row in grid[3:5]],
    ]
    for i, block in enumerate(blocks):
        if all(block != other for j, other in enumerate(blocks) if i != j):
            return [row[:] for row in block]
    raise ValueError("expected exactly one different 2x2 corner block")
```

## Generator Constraints

ARC-GEN fixes the input size to `5` and output size to `2 x 2`. The four sprites occupy rows/cols `(0:2,0:2)`, `(0:2,3:5)`, `(3:5,0:2)`, and `(3:5,3:5)`, with one blank row/column separating them. The nonzero color is a random ARC color. The shared pattern (`same`) and odd pattern (`diff`) are distinct subsets of the four sprite cells, each with size `2` or `3`; the odd corner index is sampled from `0..3`. Because `same != diff`, exactly one corner is the least-common block. There are no multicolor sprites, no noisy cells outside the four blocks, and no ambiguity in the odd-block choice.

## Reference Notes

The ARC-DSL solver computes `lefthalf/righthalf` and `tophalf/bottomhalf` to extract the four `2 x 2` corner blocks, combines them, and returns `leastcommon`. The Code Golf 2025 solution indexes the same four fixed corner blocks and picks the one with count one. These agree with ARC-GEN: the task is a fixed-position four-way odd-one-out over tiny binary masks in one foreground color.
