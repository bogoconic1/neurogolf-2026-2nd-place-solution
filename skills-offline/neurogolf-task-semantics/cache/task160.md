# task160 Semantics

## Sources

- Current champion builder: `solutions_py/task160.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task160.json`
- ARC-GEN task id: `6c434453`
- ARC-DSL task id: `6c434453`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_6c434453.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_6c434453.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task160.py`

## Pattern

The input is a 10x10 black grid containing several separated blue 3x3 sprites. Sprite type 0 is a hollow 3x3 box with eight blue cells; the other sprite types are smaller blue shapes. The output preserves all non-box blue sprites unchanged. Every hollow blue box is removed and replaced in the same 3x3 footprint by a red plus: center, up, down, left, and right cells are color 2, while the four former box corners become black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    plus = [(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)]
    box = {
        (0, 0), (0, 1), (0, 2),
        (1, 0),         (1, 2),
        (2, 0), (2, 1), (2, 2),
    }
    for r in range(h - 2):
        for c in range(w - 2):
            cells = {(dr, dc) for dr in range(3) for dc in range(3)
                     if grid[r + dr][c + dc] == 1}
            if cells != box:
                continue
            # Generator forbids adjacent/overlapping sprites, so this local
            # 3x3 test identifies exactly one hollow-box sprite.
            for dr, dc in box:
                out[r + dr][c + dc] = 0
            for dr, dc in plus:
                out[r + dr][c + dc] = 2
    return out
```

## Generator Constraints

The generator uses a square grid, normally size 10, with `num=5` blue sprites. Each sprite anchor is chosen in `[0, size-3]` for rows and columns, with type 0..5. At least one type-0 hollow box is guaranteed. The generator rejects strict 3x3 bounding-box overlaps and rejects edge-adjacent pixels from different sprites, so blue connected components are isolated. Type 0 has eight cells; the other types have two to five cells. Only colors 0, 1, and 2 appear in the output: black background, unchanged blue non-box sprites, and red replacement pluses.

## Reference Notes

The ARC-DSL solver finds connected objects, filters objects of size eight, covers those cells with black, then fills the direct-neighbor plus around each object's upper-left corner with color 2. The Code Golf solution encodes the same rewrite through a string/regex recurrence: find the hollow-box pattern and substitute the red plus pattern. The references agree; the size-eight object test is safe because the generator only gives the hollow box eight cells.
