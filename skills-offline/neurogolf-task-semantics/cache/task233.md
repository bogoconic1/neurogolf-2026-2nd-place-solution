# task233 Semantics

## Sources

- Current champion builder: `solutions_py/task233.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task233.json`
- ARC-GEN task id: `97a05b5b`
- ARC-DSL task id: `97a05b5b`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_97a05b5b.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_97a05b5b.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task233.py`

## Pattern

The input contains a large red rectangular box and one to five 3x3 colored sprite exemplars outside that box. Each exemplar has a non-red background color and red cells marking its sprite shape. Inside the red box, matching 3x3 target regions appear as black sprite cells on red/background cells, possibly rotated relative to the outside exemplar.

The output is the crop of the red box. Each inside target region is painted with the matching outside exemplar's background color, and the black sprite cells are replaced by red. Matching accounts for rotations; ambiguous tiny shapes are avoided by the generator.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(r, l={}):
    for g in range(len(r) - 2):
        for e in range(len(r[0]) - 2):
            i = (*[r[g + n // 3][e + n % 3] == 2 for n in range(9)],)
            if len((z := {r[g + n // 3][e + n % 3] for n in range(9)})) > 1 > (0 in z):
                z = {i: sum(z) - 2}
                for n in range(9):
                    r[g + n // 3][e + n % 3] = 0
                    l[i] = z
                    i = (*[i[2 + n % 3 * 3 - n // 3] for n in range(9)],)
    r = [[*n] for n in zip(*r) if sum(n)]
    r = [[*n] for n in zip(*r) if sum(n)]
    for g in range(len(r) - 2):
        for e in range(len(r[0]) - 2):
            i = (*[r[g + n // 3][e + n % 3] == 0 for n in range(9)],)
            if l.get(i, {}).get(i, {}):
                z = l.get(i, {}).pop(*l.get(i, {}))
                for n in range(9):
                    r[g + n // 3][e + n % 3] = 2 >> r[g + n // 3][e + n % 3] or z
    for g in range(len(r) - 2):
        for e in range(len(r[0]) - 2):
            i = (*[r[g + n // 3][e + n % 3] == 0 for n in range(9)],)
            if l.get(i, {}):
                z = l.get(i, {}).pop(*l.get(i, {}))
                for n in range(9):
                    r[g + n // 3][e + n % 3] = 2 >> r[g + n // 3][e + n % 3] or z
    return r


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

- Input dimensions are at most 30x30 after padding; generated width/height are red-box size plus 2..10 margin.
- Red box width and height are each 8..20.
- There are 1..5 sprites. Each outside sprite is a 3x3 patch with 4..8 red cells and a unique non-red background color.
- Sprite shapes are sampled to avoid rotation ambiguity for sparse 2x2 or 2x3-like cases.
- Inside target patches are placed within the red box with no overlap. Each target uses one of the outside sprite masks under rotation, with black cells marking the sprite.
- Outside sprite patches and the red box are non-overlapping with one-cell clearance.

## Reference Notes

- ARC-DSL extracts the largest object as the red box, forms a subgrid crop, gathers multi-color outside sprites, normalizes masks, tries mirror/rotation transforms, and paints matched colors into the crop.
- The Code Golf solution first records outside 3x3 colored patterns keyed by red mask and rotations, crops away empty rows/columns to isolate the box, then scans inside 3x3 black masks to paint matching colors.
- The task is global: the correct output at a cell depends on red-box crop location and selected sprite matches, not just a fixed local input patch.
