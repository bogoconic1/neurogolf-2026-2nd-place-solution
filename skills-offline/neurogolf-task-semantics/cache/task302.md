# task302 Semantics

## Sources

- Current champion builder: `solutions_py/task302.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task302.json`
- ARC-GEN task id: `c0f76784`
- ARC-DSL task id: `c0f76784`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_c0f76784.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_c0f76784.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task302.py`

## Pattern

The input is a fixed `12x12` grid containing up to three non-overlapping hollow gray (`5`) square frames. The possible total square sizes are `5x5`, `4x4`, and `3x3`. Each frame border is gray and each interior is black (`0`). The output keeps all gray borders unchanged and recolors each black square interior by the size of its containing frame:

- total size `3x3` has a `1x1` interior, recolored to `6`
- total size `4x4` has a `2x2` interior, recolored to `7`
- total size `5x5` has a `3x3` interior, recolored to `8`

Background outside the objects remains black.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    for size, fill in [(5, 8), (4, 7), (3, 6)]:
        for r in range(h - size + 1):
            for c in range(w - size + 1):
                ok = True
                for dr in range(size):
                    for dc in range(size):
                        border = dr in (0, size - 1) or dc in (0, size - 1)
                        expected = 5 if border else 0
                        if grid[r + dr][c + dc] != expected:
                            ok = False
                if ok:
                    for dr in range(1, size - 1):
                        for dc in range(1, size - 1):
                            out[r + dr][c + dc] = fill
    return out
```

## Generator Constraints

- Grid size is fixed at `12x12`.
- Candidate object sizes are tried in order `5`, `4`, `3`; a candidate is skipped if it overlaps prior accepted objects using `common.overlaps(..., padding=1)`.
- Thus examples may contain one, two, or three objects, but no overlapping/touching frames.
- All object borders are gray (`5`), all interiors are black in the input, and the output only changes those interiors to `8`, `7`, or `6` according to total frame size.
- There are no other colors or distractor gray structures in generated examples.

## Reference Notes

The ARC-DSL solver identifies black square objects inside the gray frames, fills all such interiors with `7`, overwrites the largest interior with `8`, and overwrites the size-one interior with `6`. This matches the generator because the only black square objects are interiors of sizes `3x3`, `2x2`, and `1x1`. The Code Golf solution uses a regex over the string form of the grid to infer each black run length between gray borders and maps interior length `k` to color `k+5`.
