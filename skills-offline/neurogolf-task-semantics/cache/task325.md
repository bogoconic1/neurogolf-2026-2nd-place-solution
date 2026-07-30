# task325 Semantics

## Sources

- Current champion builder: `solutions_py/task325.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task325.json`
- ARC-GEN task id: `d0f5fe59`
- ARC-DSL task id: `d0f5fe59`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_d0f5fe59.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_d0f5fe59.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task325.py`

## Pattern

The input contains several disconnected cyan (`8`) sprites on a black background. Each sprite is a small connected creature placed without overlap. The output is a square black grid whose side length is the number of cyan sprites. The output's main diagonal is cyan, and all off-diagonal cells are black. If there are `N` sprites, the output is `N x N` with `output[i][i] = 8` for `0 <= i < N`.

Equivalently, the transformation only needs the connected-component count of cyan objects, not their shapes or positions beyond separating components.

## Readable Python Solver

```python
def solve(grid):
    h = len(grid)
    w = len(grid[0])
    seen = [[False] * w for _ in range(h)]
    count = 0
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 8 or seen[r][c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r][c] = True
            while stack:
                rr, cc = stack.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == 8:
                        seen[nr][nc] = True
                        stack.append((nr, nc))
    out = [[0 for _ in range(count)] for _ in range(count)]
    for i in range(count):
        out[i][i] = 8
    return out
```

## Generator Constraints

ARC-GEN chooses input width and height independently from `8..16`. The number of sprites is random from roughly `(width+height)//10` to `(width+height)//5`, so public/generated examples have a small count, usually two to five. Each sprite has a bounding box side length of `3` or `4`, with a generated connected cyan creature inside it. The generator places sprite bounding boxes with at least one cell of separation using `common.overlaps(..., margin=1)`, so different sprites do not touch. All foreground pixels are cyan (`8`); the background is black.

The output shape is `num_sprites x num_sprites`, not tied to the input width or height. The only information needed is the number of disconnected cyan components.

## Reference Notes

The ARC-DSL solver calls `objects(I, T, F, T)`, takes `size(objects)` as the component count, creates a square canvas of that size, and fills the main diagonal from `ORIGIN` in direction `UNITY` with cyan. This matches the generator exactly.

The Code Golf 2025 solution is a compact recursive/counting program that effectively counts connected cyan components and then emits an `8` diagonal of that length. There is no semantic disagreement among the references.
