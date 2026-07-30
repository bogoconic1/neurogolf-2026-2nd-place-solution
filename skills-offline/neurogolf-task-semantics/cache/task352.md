# task352 Semantics

## Sources

- Current champion builder: `solutions_py/task352.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task352.json`
- ARC-GEN task id: `dc1df850`
- ARC-DSL task id: `dc1df850`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_dc1df850.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_dc1df850.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task352.py`

## Pattern

The input is a small grid with isolated colored single pixels on a black background. Red (`2`) pixels are special. The output keeps every input pixel unchanged and fills the 8-neighborhood around each red pixel with blue (`1`), leaving the red center itself red. Non-red colored pixels are not surrounded.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 2:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (dr or dc):
                            out[nr][nc] = 1
                out[r][c] = 2
    return out
```

## Generator Constraints

ARC-GEN chooses width `5..10` and height either width or width-1. Candidate pixels are chosen from a shuffled permutation of columns, so at most one object appears in each row and each selected column. A rejection test prevents selected pixels from being within Chebyshev distance `<2`, so 3x3 neighborhoods do not overlap. Selected point colors are either red or a random non-red/non-blue color. Red and blue have fixed roles: red is the source marker and blue is the added outbox color.

The NeuroGolf graph still emits a dense `[1,10,30,30]` tensor. Meaningful generated content is in the upper-left at no more than `10x10`.

## Reference Notes

ARC-DSL filters objects by color red (`2`), computes `outbox` around each red singleton, and fills those cells with blue (`1`). The original red object remains because fill only targets the outbox cells. Code Golf 2025 expresses the same local 3x3 ring rule through repeated rotations/list operations.
