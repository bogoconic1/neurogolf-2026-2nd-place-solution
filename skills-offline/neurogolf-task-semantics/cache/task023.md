# task023 Semantics

## Sources

- Current champion builder: `solutions_py/task023.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task023.json`
- ARC-GEN task id: `150deff5`
- ARC-DSL task id: `150deff5`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_150deff5.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_150deff5.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task023.py`

## Pattern

The input is an 8x9 through 9x11 grid, padded to the NeuroGolf 30x30 frame, containing gray objects on a zero background. Objects are either 2x2 gray boxes, vertical 3x1 gray sticks, or horizontal 1x3 gray sticks. The output has the same active grid size and zero background. Every box cell is recolored cyan `8`; every stick cell is recolored red `2`.

The generator prevents object overlap and prevents close parallel same-type placements that would make identical boxes or sticks ambiguous. Boxes and sticks can still be close enough that a simple local 2x2-first detector can temporarily color some stick cells as cyan; the reference solver repairs those cases by recognizing 3-cell stick patterns that contain a mix of cyan and gray after the box pass.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re

def p(g):
    l = len(g[0]) * 3 - 2
    return max([p(x) for i in ['8?5(, )5(.{%d})5(, )5' % l, '2?5' + '(.{%d}...)5' % l * 2, '2?5(, )5(, )5'] if (x := eval(i[0].join(re.split(i, (s := str(g)), 1)))) != g] + [0 ** ('5' in s) * g])


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Active width is 9, 10, or 11; active height is 8 or 9.
- If width is below 10 there are initially 2 boxes, otherwise 3 boxes; a random branch may add one more stick placement count.
- Boxes are 2x2 gray blocks. Box top-left rows are sampled from 1..5 and columns from 1..6. Boxes do not overlap and are not placed side-by-side in the same row within two columns or in the same column within two rows.
- Sticks are either vertical 3x1 or horizontal 1x3 gray objects. Stick top-left rows are sampled from 1..5 and columns from 1..6. Same-orientation sticks are not placed too close and all sticks avoid overlap with boxes and other sticks.
- Colors are fixed: background `0`, input objects gray `5`, box output cyan `8`, stick output red `2`.

## Reference Notes

- ARC-DSL first finds all 2x2 gray boxes and fills them cyan. It then finds vertical and horizontal stick patterns, including variants with two cyan cells and one gray cell, and fills those stick cells red. This staged order explains why a stick touching a detected 2x2 area can be corrected back to red.
- The Code Golf 2025 solution applies recursive regex replacements to the string form of the grid. It repeatedly replaces 2x2 box and 3-cell stick patterns until no gray remains, using the same cyan/red priority behavior as the DSL.
- Output shape equals input active shape; outside the active grid remains NeuroGolf padding/zero in the dense 30x30 output tensor.
