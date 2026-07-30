# task390 Semantics

## Sources

- Current champion builder: `solutions_py/task390.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task390.json`
- ARC-GEN task id: `f8a8fe49`
- ARC-DSL task id: `f8a8fe49`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_f8a8fe49.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_f8a8fe49.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task390.py`

## Pattern

The input is a `15x15` grid containing a red (`2`) gripper/box and gray (`5`) pixels inside the box. The red object is either horizontal or transposed vertical. The output keeps the red gripper, clears all gray pixels from inside it, and mirrors the gray pattern halves to the outside of the box: the upper half is reflected above one side of the red frame, and the lower half is reflected below the opposite side. In the transposed case the same operation is applied after transposing the grid, then transposed back.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re
p = lambda g, k=0: eval(re.sub('[^(2]{9}2' * 2 + '?', '*[\\g<0>][::-1]', f'{(*zip(*(k or p(g, g))),)}'))


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN fixes `size=15`. The non-transposed red box has width `5..7`, height `8..9`, top row `3`, and left column `2..5`. The red frame consists of full top and bottom bars plus four red corner grippers one row in from each end. Gray pixels are sampled inside the logical box from a `(wide-2) x (tall-4)` bitmap at density around `0.75`; if the height is odd the exact middle row is avoided. The top and bottom gray halves are guaranteed diagonally connected and centrally aligned. A random transpose branch swaps rows and columns for both input and output.

## Reference Notes

ARC-DSL finds the red object, determines whether it is portrait, extracts and trims the red-box subgrid, mirrors it in the orientation opposite the frame, splits it into two halves, normalizes those halves as objects, clears gray from the input, then paints the two shifted halves outside the red frame. The code-golf solution uses a regular-expression trick on the tuple representation to identify and reverse the repeated red-frame span, which is another compact expression of the same mirror-outside behavior.
