# task103 Semantics

## Sources

- Current champion builder: `solutions_py/task103.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task103.json`
- ARC-GEN task id: `44f52bb0`
- ARC-DSL task id: `44f52bb0`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_44f52bb0.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_44f52bb0.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task103.py`

## Pattern

The input is a `3x3` grid containing red (`2`) pixels on black (`0`) background.
The output is a `1x1` grid. If the input is symmetric under vertical mirroring
(left column equals right column in every row), output blue/color `1`. Otherwise
output orange/color `7`. The generator avoids ambiguous random cases where
horizontal and vertical symmetry disagree in the opposite direction; the task
rule used by ARC-DSL and the held examples is vertical mirror equality only.

## Readable Python Solver

```python
def solve(grid):
    vertical = all(grid[r][c] == grid[r][2 - c] for r in range(3) for c in range(3))
    return [[1 if vertical else 7]]
```

## Generator Constraints

ARC-GEN uses only `3x3` inputs. It samples one to four red positions, sometimes
forcing four-way symmetry by also setting the vertical, horizontal, and 180-degree
mirror positions. It rejects generated grids unless horizontal symmetry and
vertical symmetry have the same truth value, which avoids ambiguous cases. The
final output is always `1x1`, color `1` when the input equals its vertical mirror
and color `7` otherwise.

## Reference Notes

ARC-DSL computes `vmirror(I)`, compares it to `I`, branches to color `ONE` or `SEVEN`, and creates a `1x1` canvas. The Code Golf 2025 solution checks whether the first and last rows/columns mirror-equivalent under the compact Python list representation and returns `1` or `7` in a singleton grid.
