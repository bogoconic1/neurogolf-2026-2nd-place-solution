# task344 Semantics

## Sources

- Current champion builder: `solutions_py/task344.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task344.json`
- ARC-GEN task id: `d90796e8`
- ARC-DSL task id: `d90796e8`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d90796e8.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d90796e8.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task344.py`

## Pattern

Input and output have the same ARC grid size, with NeuroGolf later embedding the result in the standard `[1,10,30,30]` tensor. Colors are background `0`, red `2`, green `3`, gray `5`, and cyan `8`.

The task looks for orthogonally adjacent red/green two-cell objects. For each adjacent pair, the green cell becomes cyan `8` and the neighboring red cell becomes background `0`. Other cells are preserved: gray distractors stay gray, isolated red stays red, and any green cell without an in-bounds adjacent red stays green.

Equivalent local rule:

- a red cell becomes `0` iff at least one of its four neighbors is green;
- a green cell becomes `8` iff at least one of its four neighbors is red;
- every other color is copied unchanged.

The generator spaces red seeds by Manhattan distance greater than `2`, so intended red/green pairs are isolated from each other; a local four-neighbor test is sufficient.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v not in (2, 3):
                continue
            has_partner = False
            target = 3 if v == 2 else 2
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == target:
                    has_partner = True
                    break
            if has_partner:
                out[r][c] = 0 if v == 2 else 8
    return out
```

## Generator Constraints

- Width and height are independently sampled from `3..10`.
- Gray cells are sparse random distractors, about `4%` of the grid.
- Red seed candidates are sampled from an expanded `(width+2) x (height+2)` area and shifted by `-1`, so edge/out-of-bounds attempts can occur. Only in-bounds drawn cells appear in the final grid.
- Red seeds are accepted only when their Manhattan distance from prior accepted seeds is greater than `2`, preventing overlapping local pair neighborhoods.
- For most accepted red seeds, one green cell is drawn one step left/right/up/down; rarely no green is added. If the chosen red or green lies outside the real grid, only the in-bounds part is represented.
- The generated output starts as a copy of the input, then scans green cells and, for every adjacent red neighbor, sets that red neighbor to `0` and the green cell to `8`.

## Reference Notes

ARC-DSL collects size-2 objects whose palette contains red, removes those objects from the grid, then fills the green-side cells with cyan. The Code Golf solution rotates the grid four times and replaces textual horizontal `3, 2` occurrences with `8,0`, which is another compact way to catch every orientation of an adjacent green-red pair.

There is no disagreement between the references: both implement the same local adjacency rewrite for red/green dominoes while preserving gray noise and unrelated pixels.
