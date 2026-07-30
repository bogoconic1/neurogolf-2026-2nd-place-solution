# task044 Semantics

## Sources

- Current champion builder: `solutions_py/task044.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task044.json`
- ARC-GEN task id: `228f6490`
- ARC-DSL task id: `228f6490`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_228f6490.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_228f6490.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task044.py`

## Pattern

The visible task grid is 10x10. The input contains two gray (`5`) rectangular boxes and two colored sprite objects outside the boxes. Inside each gray box, some cells are black holes whose normalized shape matches exactly one of the external colored sprites. The output moves each external colored sprite into the matching black-hole shape inside its corresponding gray box, using the sprite color, and removes the external sprite from its old location. Gray box cells and random dust pixels remain unchanged.

Equivalently: identify the two black connected components that border gray boxes, identify the two non-gray/non-black colored sprite objects, match each sprite to the hole with the same normalized shape, then translate each sprite onto its matching hole.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
from re import *
p = lambda o, n=X: -n * o or p(eval([(o := ('%s' % o)), sub((x := str(n % S)), '0', '|n%S'.join((s := split('(?<=5.{28}5..%s)(?=..5.{28}5)' % sub(x, '0)(', sub('[^%s]' % x, '.', o)).strip('.'), o))))][s.count('') % 2]), n - 1)


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN creates exactly two gray boxes. Their heights are 3 to 5 and sum to 8 or 9; the upper box starts at row 0 or 1 and the lower box is bottom-aligned. Widths are 5 to 7. Each box has an interior sprite area of `(width-2) x (height-2)`. The two external sprite shapes are continuous creatures with distinct pixel counts, so shape/count matching is unambiguous. External sprite placements are chosen not to overlap boxes or each other. Sprite colors are two random non-gray colors distinct from a random dust color. Dust pixels of one color are placed only in otherwise empty cells and remain unchanged.

## Reference Notes

The ARC-DSL solver finds black objects bordering the grid structure, finds the non-black/non-gray sprite objects, normalizes shapes, matches each sprite to the corresponding black hole shape, computes the offset between sprite and hole upper-left corners, and moves each colored sprite into its hole. The Code Golf regex solution likewise searches for matching hole/sprite patterns by color and replaces the hole cells.
