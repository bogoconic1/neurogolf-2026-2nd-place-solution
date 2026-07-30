# task056 Semantics

## Sources

- Current champion builder: `solutions_py/task056.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task056.json`
- ARC-GEN task id: `27a28665`
- ARC-DSL task id: `27a28665`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_27a28665.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_27a28665.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task056.py`

## Pattern

The input is a 3x3 grid containing background `0` and one nonzero color. The nonzero cells form one of four fixed patterns. The output is a 1x1 grid containing the pattern id: `1`, `2`, `3`, or `6`.

The pattern can be distinguished from the two top-corner cells:

- top-left colored and top-right background -> output `1`
- top-left colored and top-right colored -> output `2`
- top-left background and top-right colored -> output `3`
- top-left background and top-right background -> output `6`

The actual nonzero input color is irrelevant.

## Readable Python Solver

```python
def solve(grid):
    a = grid[0][0] == 0
    b = grid[0][2] == 0
    if not a and b:
        return [[1]]
    if not a and not b:
        return [[2]]
    if a and not b:
        return [[3]]
    return [[6]]
```

## Generator Constraints

ARC-GEN samples one of pattern ids `(1, 2, 3, 6)` and a random nonzero color. Grid size is fixed at `3x3` in the references. Pattern `1` has colored cells `(0,0),(0,1),(1,0),(1,2),(2,1)`. Pattern `2` colors the four corners plus center. Pattern `3` colors `(0,1),(0,2),(1,1),(1,2),(2,0)`. Pattern `6` colors a plus sign centered at `(1,1)`. Validation covers multiple colors for the same pattern; color must not affect the answer.

## Reference Notes

The ARC-DSL solver counts object sizes and maps the non-background object size to ids: size `1 -> 2`, size `4 -> 3`, size `5 -> 6`, otherwise `1`. The Code Golf solution uses a shorter arithmetic expression based on the top-left and top-right cells. Both are equivalent on the generator distribution. The two-cell decoder is the more compact ONNX route.
