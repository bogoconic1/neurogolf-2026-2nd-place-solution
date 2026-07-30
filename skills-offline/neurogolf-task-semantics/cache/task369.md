# task369 Semantics

## Sources

- Current champion builder: `solutions_py/task369.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task369.json`
- ARC-GEN task id: `e8593010`
- ARC-DSL task id: `e8593010`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e8593010.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e8593010.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task369.py`

## Pattern

The input is a 10x10 grid using gray (`5`) background and black (`0`) foreground components. Each black connected component has size 1, 2, or 3. The output keeps gray cells gray, recolors size-1 black components to `3`, size-2 components to `2`, and size-3 components to `1`. The output size is unchanged.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    out = [row[:] for row in grid]
    seen = set()
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0 or (r, c) in seen:
                continue
            stack = [(r, c)]
            comp = []
            seen.add((r, c))
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr][nc] == 0 and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            color = {1: 3, 2: 2, 3: 1}[len(comp)]
            for rr, cc in comp:
                out[rr][cc] = color
    return out
```

## Generator Constraints

ARC-GEN uses a fixed square size of 10 in validation and generated examples. It starts from gray background and places non-overlapping black components. Components are isolated by 4-neighborhood gaps. Size-1 components are single cells; size-2 and size-3 components are straight horizontal or vertical bars, except the generator can produce a size-3 L-like shape by drawing a 2x2 square and deleting one corner. The hidden bitmap colors are `4 - component_size`, so the input collapses all non-gray component cells to black while the output restores the size code.

## Reference Notes

The ARC-DSL solver gets connected objects, filters size 1 and size 2, fills them with colors 3 and 2, then replaces remaining black cells with 1. Code Golf 2025 implements the same rule through repeated transposed/reversed neighbor aggregation. There is no ambiguity because components are separated and have size at most 3.
