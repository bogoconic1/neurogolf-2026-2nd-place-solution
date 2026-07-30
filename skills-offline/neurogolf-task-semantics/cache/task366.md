# task366 Semantics

## Sources

- Current champion builder: `solutions_py/task366.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task366.json`
- ARC-GEN task id: `e6721834`
- ARC-DSL task id: `e6721834`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_e6721834.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_e6721834.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task366.py`

## Pattern

The input is two equally sized panels, either side-by-side or stacked. The panels have different background colors. One panel is the source panel: it contains 2 or 3 solid foreground rectangles, all using the same foreground fill color, with a few non-foreground marker pixels inside each rectangle. The other panel is the target panel: it contains only sparse marker pixels on its background.

The output has the size of one panel and starts as the target panel background plus its marker pixels. For each source rectangle, take the marker pattern inside that rectangle after removing the common foreground fill color. If that marker pattern occurs in the target panel, copy the entire source rectangle onto the target at the matching offset, preserving both the foreground fill and marker colors. Source rectangles whose marker pattern does not occur in the target are not copied. The orientation of the two panels and which side is source are data-dependent.

## Readable Python Solver

```python
def solve(grid):
    from collections import Counter

    h, w = len(grid), len(grid[0])
    if h > w:
        panels = [[row[:] for row in grid[: h // 2]],
                  [row[:] for row in grid[h // 2 :]]]
    else:
        panels = [[row[: w // 2] for row in grid],
                  [row[w // 2 :] for row in grid]]

    # The target has fewer colors; its dominant color is its background.
    target, source = sorted(
        panels,
        key=lambda p: (len({v for row in p for v in row}),
                       -Counter(v for row in p for v in row).most_common(1)[0][1]),
    )
    ph, pw = len(target), len(target[0])
    target_bg = Counter(v for row in target for v in row).most_common(1)[0][0]
    source_bg = Counter(v for row in source for v in row).most_common(1)[0][0]

    # Extract the solid source rectangles (marker pixels remain connected to
    # their common fill). The fill is the most common non-background color.
    seen = set()
    components = []
    for sr in range(ph):
        for sc in range(pw):
            if source[sr][sc] == source_bg or (sr, sc) in seen:
                continue
            stack = [(sr, sc)]
            seen.add((sr, sc))
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < ph and 0 <= cc < pw and source[rr][cc] != source_bg and (rr, cc) not in seen:
                        seen.add((rr, cc))
                        stack.append((rr, cc))
            components.append(cells)

    fill = Counter(
        value for row in source for value in row if value != source_bg
    ).most_common(1)[0][0]
    objects = []
    for cells in components:
        r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
        c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
        rectangle = [row[c0 : c1 + 1] for row in source[r0 : r1 + 1]]
        marker_count = sum(value != fill for row in rectangle for value in row)
        objects.append((marker_count, rectangle))

    # Match the most specific marker pattern first. Painting a matched large
    # rectangle consumes its marker occurrence so one-dot patterns cannot steal it.
    out = [row[:] for row in target]
    for _, rectangle in sorted(objects, key=lambda item: item[0], reverse=True):
        rh, rw = len(rectangle), len(rectangle[0])
        found = False
        for top in range(ph - rh + 1):
            for left in range(pw - rw + 1):
                expected = [
                    [target_bg if value == fill else value for value in row]
                    for row in rectangle
                ]
                if all(out[top + r][left : left + rw] == expected[r] for r in range(rh)):
                    for r in range(rh):
                        out[top + r][left : left + rw] = rectangle[r][:]
                    found = True
                    break
            if found:
                break
    return out
```

## Generator Constraints

- Base panel height is 10 through 15; width is height plus -2 through +2.
- Input panels are either horizontal or vertical concatenations of the same base size.
- Two background colors are sampled, plus one common foreground fill color distinct from both backgrounds.
- There are 2 boxes when height is less than 12, otherwise 3 boxes. Box widths/heights are 2 through 7.
- Every source box has `idx + 1` marker pixels, all the same marker color for that box. Marker colors are distinct from both backgrounds and the foreground fill, but different boxes may share a marker color.
- Source boxes are non-overlapping within a panel. The target panel may contain only a subset of the box marker patterns.
- The output is only the target panel size, not the full input size.

## Reference Notes

The ARC-DSL solution splits the input by orientation, orders the two panels by number of colors, treats the fewer-color panel as the target, extracts source objects from the richer panel, identifies the common foreground fill color, filters each object down to its non-foreground marker pixels, finds marker-pattern occurrences in the target, shifts matching full source objects to those occurrences, and paints them on the target.

The Code Golf solution uses the same idea in compressed form: choose orientation and side from panel dimensions/colors, enumerate candidate source rectangles in descending size/order, match their marker pixels against the target, then repaint the aligned rectangle.
