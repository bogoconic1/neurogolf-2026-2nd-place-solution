# task173 Semantics

## Sources

- Current champion builder: `solutions_py/task173.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task173.json`
- ARC-GEN task id: `72322fa7`
- ARC-DSL task id: `72322fa7`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_72322fa7.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_72322fa7.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task173.py`

## Pattern

The grid contains several non-overlapping copies of one to three 3x3 sprite types. Each sprite type uses a unique pair of nonzero colors: an outer color and a center color. The possible sprite shapes are:

- X: outer pixels at the four corners, center at `(1,1)`.
- Plus: outer pixels at top/middle/bottom/left/right arms, center at `(1,1)`.
- Horizontal line: outer pixels at `(1,0)` and `(1,2)`, center at `(1,1)`.
- Vertical line: outer pixels at `(0,1)` and `(2,1)`, center at `(1,1)`.

For each sprite type, exactly one copy is fully visible in the input. Other copies are partial: either the center pixel is hidden, leaving only outer-color pixels, or the outer pixels are hidden, leaving only the center-color pixel. The output restores all missing pixels so every copy becomes the complete sprite matching its type's full exemplar. Background and already visible pixels are preserved.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
p = lambda f: [(e := [f[a - n // 3][i - n % 3] for n in range(9)]) == e[::-1] * any(e[:sum(((f[c - n // 3][m - n % 3] == e[n]) << 3 * (n == 4) for n in range(9))) // 8 * 4] * e[4]) == exec('for n in range(9):f[c-n//3][m-n%3]=e[n]') for c in range(len(f)) for a in range(len(f)) for m in range(len(f[0])) for i in range(len(f[0]))] and f


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Width and height are each random from 10 to 25; NeuroGolf embeds them in the standard 30x30 tensor.
- There are 1 to 3 sprite types.
- For each sprite type, two distinct colors are drawn from a shuffled list of colors 1..9, so color pairs are unique and no foreground color is reused across sprite types.
- Sprite type is one of X, plus, horizontal, or vertical.
- Each type has 2 to 4 copies. The first copy for that type is full (`megashow=0`); every later copy hides either the center (`megashow=1`) or the outer pixels (`megashow=2`).
- Sprite top-left rows/cols are chosen so the 3x3 windows fit in the grid and do not overlap, with a separation constraint from `common.overlaps(..., 2)`.

## Reference Notes

The ARC-DSL solver separates foreground objects, uses single-color partial objects versus multi-color full objects, finds occurrences of each visible component pattern, and paints shifted normalized full objects over matching partial locations. The Code Golf solution scans 3x3 windows and checks reversal/symmetry-style sprite signatures, then writes the complete matched 3x3 sprite into every matching partial window.
