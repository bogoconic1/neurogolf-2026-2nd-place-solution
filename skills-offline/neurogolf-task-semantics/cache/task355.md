# task355 Semantics

## Sources

- Current champion builder: `solutions_py/task355.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task355.json`
- ARC-GEN task id: `de1cd16c`
- ARC-DSL task id: `de1cd16c`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_de1cd16c.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_de1cd16c.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task355.py`

## Pattern

The input is a rectangular grid partitioned into large solid-color rectangles. A small number of single-cell marker pixels, all in one marker color, overwrite cells inside those rectangles. The marker color is the globally least frequent color. For each large non-marker object/rectangle, count how many marker cells lie inside that object. Return a `1x1` grid whose single value is the dominant/base color of the object with the largest marker count.

The output is always one color cell. ARC-GEN samples marker counts without replacement from `0..5`, so the winning original rectangle has a unique maximum marker count. If adjacent generated rectangles share a color, ARC-DSL may merge them as one object, but the output color is still the shared dominant color.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])
    vals = [v for row in grid for v in row]
    marker = min(set(vals), key=vals.count)

    seen = [[False] * w for _ in range(h)]
    best_count = -1
    best_color = None
    for r0 in range(h):
        for c0 in range(w):
            color = grid[r0][c0]
            if color == marker or seen[r0][c0]:
                continue
            stack = [(r0, c0)]
            seen[r0][c0] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == color:
                        seen[nr][nc] = True
                        stack.append((nr, nc))

            if len(cells) == 1:
                continue
            rs = [r for r, _ in cells]
            cs = [c for _, c in cells]
            marker_count = sum(
                grid[r][c] == marker
                for r in range(min(rs), max(rs) + 1)
                for c in range(min(cs), max(cs) + 1)
            )
            if marker_count > best_count:
                best_count = marker_count
                best_color = color

    return [[best_color]]
```

A compact equivalent for this generator is: for each candidate base color `c`, count marker-colored cells whose row contains `c` and whose column contains `c`; choose the color with the largest count.

## Generator Constraints

ARC-GEN draws two column widths in `5..10` and usually two row heights in `5..10`; validation includes a three-row by two-column partition. The total grid stays well under the NeuroGolf `30x30` padded envelope. Each generated rectangle receives a base color sampled from `0..9`; repeated base colors are allowed. The marker color is sampled outside the chosen base colors. For each rectangle, the number of overwritten marker pixels is sampled without replacement from `range(6)`, so counts are small (`0..5`) and unique across generated rectangles. Marker cells are arbitrary positions inside their rectangle.

The ARC examples include black as a base color and repeated base colors, so implementations cannot treat zero as background or assume all generated rectangles have distinct colors. The input and output in NeuroGolf are one-hot encoded/padded to `[1,10,30,30]` and `[1,10,30,30]` respectively even though the logical output is `1x1`.

## Reference Notes

ARC-DSL computes the least color, removes singleton objects, builds subgrids for the large objects, selects the subgrid with the highest least-color count, and returns the most common color in that subgrid. This confirms the marker is identified by rarity, not by a fixed color.

The Code Golf 2025 solution searches candidate colors and counts cells whose value is not that candidate color while the candidate color appears in both the same row and column. On this generator that counts marker pixels inside the candidate-colored rectangle without explicitly finding boundaries. It also searches counts descending from `5`, matching the generator bound on marker counts.
