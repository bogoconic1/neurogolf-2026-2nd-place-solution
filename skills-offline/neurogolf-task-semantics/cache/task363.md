# task363 Semantics

## Sources

- Current champion builder: `solutions_py/task363.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task363.json`
- ARC-GEN task id: `e5062a87`
- ARC-DSL task id: `e5062a87`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e5062a87.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e5062a87.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task363.py`

## Pattern

The input is a 10x10 grid whose background is black (`0`) and gray (`5`). A small diagonally connected sprite pattern appears several times. One occurrence is red (`2`) in the input; the other occurrences are black holes in the gray/black background. The output paints every occurrence of that same sprite red, preserving all other cells.

Equivalently: read the red sprite shape, find all placements where that shape fits exactly on black cells in the input, and recolor those cells red. The generator makes the intended black placements exhaustive, so no extra accidental black placement should be painted.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
from re import *

def p(g):
    h = hash((*g[3],))
    g[~h % 7][3] |= h % 149 < 1
    return eval(sub(*'10', eval("'2'.join(split(sub('2',')0(',sub('[^2]','.',K:=str(g))).strip('.()')," * 3 + 'K))))))')))


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Grid size is always 10x10.
- Background cells are black (`0`) or gray (`5`).
- The sprite bounding box has `wide` from `1..4` and `tall = 5 - wide`, so its box fits within 4x4.
- A random number of cells may be removed from the sprite, but the remaining sprite is diagonally connected.
- There are `2..5` sprite placements.
- All placements are drawn black first; then all output placements are painted red.
- In the input, only the first placement is painted red; the remaining placements stay black.
- The generator rejects cases where the same black sprite could be placed somewhere that was not one of the planned placements.

## Reference Notes

The ARC-DSL solver extracts the red cells, recolors that pattern to black, finds all occurrences of that black pattern in the input, shifts the normalized red sprite to those occurrences, filters valid anchors, merges them, recolors the merged cells red, and paints them back onto the input.

The Code Golf 2025 solution is a compact regular-expression/hash trick over stringified grids. It relies on the same idea: infer the red pattern and stamp it onto matching black copies.
