# task271 Semantics

## Sources

- Current champion builder: `solutions_py/task271.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task271.json`
- ARC-GEN task id: `ae4f1146`
- ARC-DSL task id: `ae4f1146`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ae4f1146.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ae4f1146.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task271.py`

## Pattern

The input is a fixed `9x9` grid containing four non-overlapping `3x3` mini-boxes. Each mini-box is filled with cyan (`8`) background and some blue (`1`) cells. The boxes may touch at corners but not along edges. The output is the `3x3` mini-box with the largest number of blue cells, preserving blue cells and cyan background. Because the generator samples four distinct blue-cell counts and sorts them before assigning box indices, the selected box is uniquely the fourth box in the generated latent order, but it can appear anywhere in the `9x9` input.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    boxes = []
    for r in range(h - 2):
        for c in range(w - 2):
            block = [row[c:c + 3] for row in grid[r:r + 3]]
            if all(v in (1, 8) for row in block for v in row):
                if r > 0 and all(grid[r - 1][cc] in (1, 8) for cc in range(c, c + 3)):
                    continue
                if c > 0 and all(grid[rr][c - 1] in (1, 8) for rr in range(r, r + 3)):
                    continue
                boxes.append((sum(v == 1 for row in block for v in row), block))
    return max(boxes, key=lambda item: item[0])[1]
```

## Generator Constraints

- Input size is fixed at `9x9`; output size is fixed at `3x3`.
- There are exactly four `3x3` boxes. Every cell inside each box is initially cyan (`8`), then a subset of cells is changed to blue (`1`).
- The four boxes are placed so they do not overlap and do not touch along edges. Corner-touching is allowed.
- The four blue-cell counts are distinct values sampled from `0..8` and sorted. The generated output is the box associated with the largest count.
- Blue subsets are arbitrary positions within each `3x3` box.

## Reference Notes

The ARC-DSL solver finds objects with diagonal connectivity allowed, scores each object by its count of color `1`, and returns the subgrid of the object with the largest blue count. ARC-GEN confirms the object geometry is always a `3x3` cyan/blue mini-box and that the intended winner is unique. The Code Golf solution's nested `zip` trick effectively enumerates `3x3` windows and chooses the one whose string representation contains the most `1` values.
