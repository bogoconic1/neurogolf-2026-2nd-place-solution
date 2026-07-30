# task008 Semantics

## Sources

- Current champion builder: `solutions_py/task008.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task008.json`
- ARC-GEN task id: `05f2a901`
- ARC-DSL task id: `05f2a901`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_05f2a901.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_05f2a901.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task008.py`

## Pattern

The input contains a black background, one red object (color 2), and one cyan 2x2 block (color 8). The output keeps the cyan block fixed and translates the entire red object in a straight horizontal or vertical line until it is adjacent to the cyan block. The red object shape, including any missing black nibbles inside its bounding rectangle, is preserved exactly. The output size is the same as the input size.

The generated canonical orientation places the red object above the cyan block with a one-axis gap; the task may then be vertically flipped and/or transposed, so the cyan block may be above, below, left, or right of the red object. The required displacement is the shortest axis-aligned displacement that makes the moved red object touch the cyan 2x2 block without overlapping it.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    red = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 2]
    cyan = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 8]

    rr0 = min(r for r, _ in red); rr1 = max(r for r, _ in red)
    rc0 = min(c for _, c in red); rc1 = max(c for _, c in red)
    cr0 = min(r for r, _ in cyan); cr1 = max(r for r, _ in cyan)
    cc0 = min(c for _, c in cyan); cc1 = max(c for _, c in cyan)

    dr = dc = 0
    if rr1 < cr0:
        dr = cr0 - rr1 - 1
    elif cr1 < rr0:
        dr = cr1 - rr0 + 1
    elif rc1 < cc0:
        dc = cc0 - rc1 - 1
    elif cc1 < rc0:
        dc = cc1 - rc0 + 1

    out = [[0 for _ in range(w)] for _ in range(h)]
    for r, c in cyan:
        out[r][c] = 8
    for r, c in red:
        out[r + dr][c + dc] = 2
    return out
```

## Generator Constraints

- Width and height are each between 8 and 16.
- Red starts as a rectangle of width 4 or 5 and height 2 or 3, with selected cells removed as black nibbles. The nibbles are constrained so they do not erase an entire row or column and do not simply form a smaller rectangle.
- Cyan is always a solid 2x2 block.
- In the base orientation red is above cyan, with `delta = cyanrow - redrow - tall`; output shifts red down by this delta. Optional vertical flip and optional transpose create all four axis-aligned relative directions.
- The cyan column is chosen near the red columns so the moving object and cyan block are aligned for a single-axis gravity move.
- Colors are fixed: black 0, red 2, cyan 8.

## Reference Notes

ARC-DSL expresses the rule directly as `gravitate(red_object, cyan_object)` followed by `move(input, red_object, displacement)`. ARC-GEN shows the exact size bounds, 2x2 cyan block, red rectangle dimensions, nibble constraints, and flip/transpose branches. The Code Golf solution sorts rows/columns recursively; it encodes the same idea that the object should slide along an axis until the cyan block controls the ordering/contact.
