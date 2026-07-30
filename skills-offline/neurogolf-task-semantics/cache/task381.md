# task381 Semantics

## Sources

- Current champion builder: `solutions_py/task381.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task381.json`
- ARC-GEN task id: `ef135b50`
- ARC-DSL task id: `ef135b50`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ef135b50.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ef135b50.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task381.py`

## Pattern

Input is a 10x10 grid using black `0` and red `2`. The red cells form 3 to 5 axis-aligned rectangles. The output keeps all original red cells and paints maroon `9` into black cells that lie horizontally between red cells in the same row, but only for rows where that between-red run is not adjacent vertically to red above or below. Equivalently, scan each row left to right: after leaving a red run, black cells before the next red run become `9` if none of those black cells touches red in the row above or below; otherwise the gap stays black. Top and bottom rows are constrained to have no maroon fill.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]

    for r in range(h):
        last_red = w
        danger = False
        for c in range(w):
            if out[r][c] != 2:
                above_red = r > 0 and out[r - 1][c] == 2
                below_red = r + 1 < h and out[r + 1][c] == 2
                danger = danger or above_red or below_red
                continue

            if last_red < c - 1 and not danger:
                for cc in range(last_red + 1, c):
                    out[r][cc] = 9
            last_red = c
            danger = False

    return out
```

## Generator Constraints

ARC-GEN always uses `size=10`, black background, and red rectangles. Random instances choose 3 to 5 boxes, widths 1 to 4, heights 2 to 6, and legal top-left coordinates inside the 10x10 canvas. Rectangles are rejected until they do not overlap within a one-cell margin and until the generated output has no maroon in the first or last row. A further rejection rule ensures there is no remaining open black gap between red cells on interior rows. The known examples include rectangles touching borders, width-1 rectangles, and vertically staggered boxes.

## Reference Notes

The ARC-DSL solver forms pairs of red cells in the same row, connects each pair, intersects those horizontal segments with black cells, fills them with `9`, then trims/shifts/paints to account for the generator's one-cell vertical-danger condition. The ARC-GEN implementation is the clearest source for the danger logic: a black cell between two red encounters blocks maroon fill if it touches red immediately above or below. The Code Golf 2025 solution encodes the same row scan compactly. There is no disagreement between the references; ARC-DSL is more abstract while ARC-GEN states the rejection constraints explicitly.
