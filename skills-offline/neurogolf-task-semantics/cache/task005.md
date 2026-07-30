# task005 Semantics

## Sources

- Current champion builder: `solutions_py/task005.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task005.json`
- ARC-GEN task id: `045e512c`
- ARC-DSL task id: `045e512c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_045e512c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_045e512c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task005.py`

## Pattern

The grid is always 21x21. A complete 3x3 sprite appears near the center in one special color. Two or three smaller colored marker objects appear exactly one 4-cell step away from the central sprite in sampled compass directions. The output keeps the input and stamps recolored copies of the complete central sprite repeatedly every 4 cells along each marker direction until the copies leave the grid. Each ray uses the marker object's color.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
import re

def p(g):
    c = max('987643251', key=str(g).count)
    return [(g := eval(re.sub(f"(?=(\\d{'.' * n})*{c}.*([^0]){'.' * n + c})0", '\\2', f'{(*zip(*g[::-1]),)}#' * 2))) for n in [11, 271, 0] * 4][-1]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN uses a fixed 21x21 grid. The central sprite is a 3x3 mask: it starts as full 3x3, may remove the center, may remove one side cell, or may remove one or both diagonal pairs. The central top-left coordinate is sampled from rows and columns 6..12, so repeated 4-cell steps can extend in several directions. The generator samples two or three nonzero compass directions, excluding the direction corresponding to a removed side cell when applicable. The first copy in each non-central direction appears only as a small leading marker in the input; all repeated copies are fully painted in the output. Colors are nonzero; the central sprite color is distinct from marker colors, while marker colors may repeat.

## Reference Notes

The ARC-DSL solver finds 8-connected colored objects, chooses the largest object as the central sprite, removes it from the object set, and for every remaining marker object computes its relative `position` to the central sprite. It shifts the central sprite by that direction multiplied by 4, 8, 12, and 16, recolors those shifted copies with the marker color, and paints them on top of the input. The Code Golf solution encodes the same repeated 4-step stamping by applying row/column transformations repeatedly.
