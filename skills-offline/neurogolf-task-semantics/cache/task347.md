# task347 Semantics

## Sources

- Current champion builder: `solutions_py/task347.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task347.json`
- ARC-GEN task id: `dae9d2b5`
- ARC-DSL task id: `dae9d2b5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_dae9d2b5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_dae9d2b5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task347.py`

## Pattern

The input is a `3x6` grid made from two side-by-side `3x3` halves. The left
half contains yellow (`4`) pixels on black; the right half contains green (`3`)
pixels on black. The output is a `3x3` mask: every cell that was yellow in the
left half or green in the corresponding right-half cell becomes pink (`6`), and
all other cells stay black.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(3)] for _ in range(3)]
    for r in range(3):
        for c in range(3):
            if grid[r][c] == 4 or grid[r][c + 3] == 3:
                out[r][c] = 6
    return out
```

## Generator Constraints

ARC-GEN uses fixed `size=3`, so inputs are always height `3` and width `6`; the
standard NeuroGolf tensor still pads this into `[1,10,30,30]`. Random examples
choose at least one nonempty set of pixels for color `4` in the left half and at
least one nonempty set for color `3` in the right half. The sets may overlap in
projected `3x3` coordinates. Validation examples exercise different sparse and
dense unions across the two halves.

## Reference Notes

The ARC-DSL solver splits the grid into left and right halves, takes positions of color `4` in the left half and color `3` in the right half, combines those coordinate sets, and fills the left-half canvas with color `6` at the combined positions. The Code Golf 2025 solution is a compact row-wise expression for the same OR-mask over paired cells.
