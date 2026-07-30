# task315 Semantics

## Sources

- Current champion builder: `solutions_py/task315.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task315.json`
- ARC-GEN task id: `cce03e0d`
- ARC-DSL task id: `cce03e0d`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_cce03e0d.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_cce03e0d.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task315.py`

## Pattern

The input is a fixed 3x3 grid with background `0` and foreground colors `1` and `2`. The output is a 9x9 block matrix. For each input cell `(br, bc)`, if that cell is red/color `2`, the corresponding 3x3 output block `(3*br:3*br+3, 3*bc:3*bc+3)` is a copy of the whole input grid. If the selector cell is background `0` or color `1`, that output block is all zero.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(9)] for _ in range(9)]
    for br in range(3):
        for bc in range(3):
            if grid[br][bc] == 2:
                for r in range(3):
                    for c in range(3):
                        out[3 * br + r][3 * bc + c] = grid[r][c]
    return out
```

## Generator Constraints

ARC-GEN fixes input size `3` and output size `9`. It samples between 2 and 8 cells from the 3x3 grid and assigns each sampled cell color `1` or `2`; all unsampled cells are `0`. Output blocks are generated only for input cells whose color is `2` (`common.red()`); each such block copies every sampled input color, including both `1` and `2`. Multiple red selector cells can therefore create multiple copies of the same 3x3 input pattern.

## Reference Notes

The ARC-DSL solver builds a 3x3 tiling of the input, then zeros all cells that correspond to positions where the 3x-upscaled selector grid is color `0` or color `1`; only color-`2` selector blocks survive. The Code Golf solution `y & -x % 5` is the same trick: selector value `2` yields mask `3`, preserving input values `1` and `2`; selector values `0` or `1` produce masks that zero those values.
