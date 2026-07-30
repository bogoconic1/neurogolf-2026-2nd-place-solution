# task317 Semantics

## Sources

- Current champion builder: `solutions_py/task317.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task317.json`
- ARC-GEN task id: `ce22a75a`
- ARC-DSL task id: `ce22a75a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ce22a75a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ce22a75a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task317.py`

## Pattern

The input is a fixed 9x9 grid, conceptually split into a 3x3 array of 3x3 blocks. Each selected block has a single gray marker (`5`) at its center coordinate `(3*r+1, 3*c+1)`. The output is also 9x9: every 3x3 block whose center had a gray marker is filled entirely with blue (`1`), and all unselected blocks remain background `0`.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(9)] for _ in range(9)]
    for br in range(3):
        for bc in range(3):
            if grid[3 * br + 1][3 * bc + 1] == 5:
                for dr in range(3):
                    for dc in range(3):
                        out[3 * br + dr][3 * bc + dc] = 1
    return out
```

## Generator Constraints

ARC-GEN fixes `size=3`, so the input/output shape is always 9x9. It samples a nonempty subset of positions from a 3x3 macro-grid. For each selected macro-cell `(r,c)`, the input places gray (`5`) only at the center of that 3x3 block. The output fills the entire corresponding 3x3 block with blue (`1`). Background is `0`; no other colors, shapes, sizes, rotations, or tie cases occur.

## Reference Notes

The ARC-DSL solver finds objects, takes each object's `outbox`, expands to the `backdrop`, and fills those cells with color `1`; because the only objects are isolated gray center pixels, this exactly maps each marker to its surrounding 3x3 block. The Code Golf solution encodes the same recursive idea: when it sees a gray center marker, emit three copies/rows of blue for the corresponding block.
