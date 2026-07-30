# task101 Semantics

## Sources

- Current champion builder: `solutions_py/task101.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task101.json`
- ARC-GEN task id: `447fd412`
- ARC-DSL task id: `447fd412`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_447fd412.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_447fd412.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task101.py`

## Pattern

The input contains two to four placed copies of the same small connected sprite.
The first copy has magnification 1 and shows the full sprite: most pixels are
blue and exactly two marker pixels are red. Every later copy is drawn at
magnification 1, 2, or 3, but in the input only the red marker pixels are shown;
its blue pixels are omitted. The output is the same canvas with the missing blue
pixels restored in every later copy at that copy's magnification. Red marker
pixels remain red, and the original full reference copy is preserved.

## Readable Python Solver

The previous cache implementation did not survive replay. The compact executable oracle below is the bundled Code Golf 2025 reference; the Pattern section remains the readable specification.

```python
def p(r):
    t, *d = (len(r[0]) + 2,)
    for a in r:
        d += (*a, 0, 0)
    f = [d.index(1)]
    for a in f:
        f += [a for a in (a - t, a - 1, a + 1, a + t) if len(d) > a > (a in f) * t < d[a]]
    e = f
    for i in range(len(d)):
        f = [i]
        for a in f:
            f += [a for a in (a - t, a - 1, a + 1, a + t) if len(d) > a > (a in f) * t < d[a]]
        l = 58 % len(f) + 2 >> 1
        for a in e:
            for p in range(l * l * all((d[a] > 1 for a in f))):
                d[i + l * a - l * min((a for a in e if d[a] > 1)) + p // l * t + p % l] = d[a]
    return [d[i * t:][:t - 2] for i in range(len(r))]


def solve(grid):
    result = p([list(row) for row in grid])
    return [list(row) for row in result]
```

## Generator Constraints

ARC-GEN creates `num_sprites` in `[2, 4]`. For two or three sprites the canvas is
12x14; for four sprites it is 14x17, although reference validation also includes
a custom 21x17 case. The base sprite width and height are each 3 or 4. The first
placed copy always has magnification 1 and contains the complete sprite. Later
copies use magnification 1, 2, or 3. Placements are regenerated until the scaled
bounding boxes do not overlap with a one-cell margin. The sprite is a connected
creature with about half of the base bounding-box pixels filled. Its pixels are
blue except for two randomly selected red marker pixels. In the input, later
copies draw only the red pixels; in the output, later copies include the full
scaled blue and red sprite.

## Reference Notes

ARC-DSL selects the object with the maximum number of colors as the reference, normalizes it, separates the minority non-blue marker cells from the blue body, searches the input for occurrences of the marker pattern, and paints shifted upscaled copies of the reference for scales 1, 2, and 3. Code Golf 2025 uses a compact flood-fill and marker-occurrence scan, but follows the same idea: infer the full creature from the only two-color object and replicate it at matching red-marker placements.
