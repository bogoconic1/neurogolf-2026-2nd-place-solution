# task396 Semantics

## Sources

- Current champion builder: `solutions_py/task396.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task396.json`
- ARC-GEN task id: `fcb5c309`
- ARC-DSL task id: `fcb5c309`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_fcb5c309.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_fcb5c309.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task396.py`

## Pattern

The input is a black grid containing two or three separated rectangular frames in one nonzero box color, plus sparse pixels in a second nonzero marker color. The marker color is the least frequent non-background color. Select the largest non-marker object, which is the largest rectangular frame. Output the cropped bounding box of that frame, preserving black interior cells and any marker pixels inside the crop, but recolor every cell of the frame color to the marker color. Logical output size is the selected rectangle size; in NeuroGolf the dense model output is the standard `[1, 10, 30, 30]` encoding with the crop placed at the origin.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter, deque

    h, w = len(grid), len(grid[0])
    counts = Counter(v for row in grid for v in row if v != 0)
    marker = min(counts, key=counts.get)
    box_colors = [c for c in counts if c != marker]
    box_color = max(box_colors, key=lambda c: counts[c])

    seen = [[False] * w for _ in range(h)]
    best = []
    for r in range(h):
        for c in range(w):
            if seen[r][c] or grid[r][c] != box_color:
                continue
            comp = []
            q = deque([(r, c)])
            seen[r][c] = True
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr][nc] and grid[nr][nc] == box_color:
                        seen[nr][nc] = True
                        q.append((nr, nc))
            if len(comp) > len(best):
                best = comp

    r0 = min(r for r, _ in best)
    r1 = max(r for r, _ in best)
    c0 = min(c for _, c in best)
    c1 = max(c for _, c in best)
    out = []
    for r in range(r0, r1 + 1):
        row = []
        for c in range(c0, c1 + 1):
            v = grid[r][c]
            row.append(marker if v == box_color else v)
        out.append(row)
    return out
```

## Generator Constraints

ARC-GEN uses input width and height independently in `[12, 18]`. It chooses two nonzero colors, with one box/frame color and one marker/noise color, then creates two or three non-overlapping boxes. Box widths and heights are sampled from `3..8`, sorted descending, so the first generated box is the unique largest by both dimensions and is the output source. Each box is a solid border in the box color with black interior. For each box, `0..4` interior marker pixels may be inserted, capped by interior area. Additional marker-color static pixels are sprinkled over about five percent of the full grid, including outside all boxes. Boxes are generated with a one-cell non-overlap margin. Output is exactly the first/largest box crop with the border recolored to the marker color.

## Reference Notes

ARC-DSL computes `leastcolor(I)`, removes least-color objects, takes the largest remaining object, crops its subgrid, then replaces that object color with the least color. This confirms that sparse marker pixels outside the box should not affect the crop except through marker-color selection. The Code Golf 2025 solution uses string/count tricks to identify the two relevant colors and flips/rotates the grid to crop the selected rectangle, consistent with the largest-box crop rule. There is no semantic disagreement between the references and the generator.
