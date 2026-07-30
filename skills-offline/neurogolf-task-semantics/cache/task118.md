# task118 Semantics

## Sources

- Current champion builder: `solutions_py/task118.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task118.json`
- ARC-GEN task id: `50846271`
- ARC-DSL task id: `50846271`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_50846271.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_50846271.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task118.py`

## Pattern

The input is a noisy gray field containing several red plus signs. In the true output, any gray static cell that lies underneath a red plus arm is cyan. The input hides those cyan cells by showing them as gray, so the task is to reconstruct the complete plus signs and recolor the hidden gray-on-plus cells to cyan while leaving ordinary gray noise gray and preserving visible red cells.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda o, n=23, d=2: -n * o or p(eval(re.sub(['(?<=[^)05], )5(?=[^)]{,8}2)', '5(?=.{,%d}[^)05], [^)05].{%d}[^)05]|..{%d}[^)05]{%d}|[^)05]{%d}..( |0|.{15}8, 8))' % (3 * d, 3 * len(o) - 2, 3 * len(o) - 2 - 3 * d, 3 + 6 * d, 6 * d)][n < 12], '8', f'{(*zip(*o[::-1]),)}')), n - 1, d | ('p' in re.sub(20 * '[^)05]', 'p', f'{o}')))


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Active grid height is 10..25 and width is height plus -3..3.
- Background is black 0; random static gray cells use color 5.
- Four non-overlapping plus signs are attempted, with arm length either 2 or 3.
- Plus signs are drawn in red 2. If a plus arm cell overlaps existing gray static, the true output cell is cyan 8; the input hides it as gray 5.
- Cross centers are at least `length` cells from the border and no closer than two cells under the generator's overlap test.
- The output differs from the input only by changing hidden gray-on-plus cells from 5 to 8.

## Reference Notes

- ARC-DSL first connects visible red cells into long horizontal/vertical red line segments, fills those segments red to reconstruct the plus arm support, measures red-object width to infer half-length, then chooses likely hidden cells by local red-neighbor counts and fills connecting segments cyan. It finally restores original visible red cells.
- Code Golf repeatedly applies regex-like substitutions over flattened and transposed grid views to turn the hidden gray cells into cyan.
