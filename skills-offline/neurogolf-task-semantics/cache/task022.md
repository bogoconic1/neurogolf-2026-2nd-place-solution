# task022 Semantics

## Sources

- Current champion builder: `solutions_py/task022.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task022.json`
- ARC-GEN task id: `137eaa0f`
- ARC-DSL task id: `137eaa0f`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_137eaa0f.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_137eaa0f.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task022.py`

## Pattern

The input is an 11x11 grid containing up to four sparse 3x3 fragments placed far enough apart that their neighborhoods do not overlap. Each nonzero fragment has a gray center pixel and two same-colored pixels in distinct non-center positions of the surrounding 3x3 neighborhood. One fragment color may be zero; that fragment is invisible because both its center and its two cells remain background zero.

The output is always a 3x3 grid. The center is gray. For each of the eight non-center positions, copy the nonzero color that appears at that relative offset from one of the gray centers. If no nonzero fragment contributes that relative offset, the output cell stays zero. The eight non-center cells are partitioned into four color pairs by the generator.

## Readable Python Solver

```python
def solve(grid):
    out = [[0 for _ in range(3)] for _ in range(3)]
    out[1][1] = 5
    h, w = len(grid), len(grid[0])
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 5:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        v = grid[rr][cc]
                        if v:
                            out[dr + 1][dc + 1] = v
    return out
```

## Generator Constraints

- Input size is fixed at 11x11 and output size is fixed at 3x3.
- The eight non-center positions of the 3x3 output are shuffled, then assigned to four fragment indices with exactly two positions per fragment.
- Four distinct colors are sampled from 1 through 9, with sampled color 5 remapped to zero. Thus the visible colors are distinct, and at most one fragment is invisible zero.
- Fragment centers are chosen from rows/cols 1 through 9 and must not be within both 4 rows and 4 columns of a previous center, preventing 3x3 neighborhood overlap.
- For a nonzero fragment, the center is gray 5 and its two assigned local positions are set to that fragment color. For a zero fragment, the center and its assigned local positions remain zero, so the corresponding output cells are zero.

## Reference Notes

- ARC-DSL finds foreground objects, filters each object for its gray center, shifts the object so that the gray center lands at the 3x3 center, and paints all shifted objects onto a zero 3x3 canvas.
- The Code Golf 2025 solution uses a compact flatten-and-shift trick. For each relative offset, it pairs every input value with the value at that offset, sorts pairs, and takes dictionary key 5. Because tuple sorting keeps the largest shifted value last among gray-center keys, this recovers the nonzero color at that offset, or zero when the offset belongs to the invisible fragment.
- The references agree that gray is the alignment anchor and that zeros are meaningful output cells, not missing data.
