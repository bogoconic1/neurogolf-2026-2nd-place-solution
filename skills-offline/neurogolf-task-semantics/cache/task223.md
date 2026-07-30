# task223 Semantics

## Sources

- Current champion builder: `solutions_py/task223.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task223.json`
- ARC-GEN task id: `9172f3a0`
- ARC-DSL task id: `9172f3a0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_9172f3a0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_9172f3a0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task223.py`

## Pattern

The input is a 3x3 grid embedded in the standard NeuroGolf input tensor. It contains 4 or 5 non-black cells drawn from 2 or 3 non-black colors; black cells remain black. The output is the same grid upscaled by a factor of 3 in both axes: every input cell becomes a solid 3x3 block of the same color. The upscaled 9x9 result occupies the top-left of the standard `[1, 10, 30, 30]` output tensor, and all cells outside that 9x9 area are black/empty.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(len(grid[0]) * 3)] for _ in range(len(grid) * 3)]
    for r, row in enumerate(grid):
        for c, color in enumerate(row):
            for dr in range(3):
                for dc in range(3):
                    out[3 * r + dr][3 * c + dc] = color
    return out
```

## Generator Constraints

ARC-GEN uses fixed `size=3`. It samples 4 or 5 distinct pixel positions from the 3x3 grid. It chooses 2 or 3 random non-black colors, then assigns each sampled pixel one of those colors by index. Unselected cells are black. There are no geometric branches beyond the pure upscale; repeated colors are allowed through repeated color indices.

## Reference Notes

The ARC-DSL solver is exactly `upscale(I, THREE)`. The Code Golf solution recursively repeats each row and each cell three times, confirming that the transformation is nearest-neighbor 3x upscaling and not object-specific logic.
