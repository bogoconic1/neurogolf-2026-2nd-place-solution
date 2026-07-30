# task379 Semantics

## Sources

- Current champion builder: `solutions_py/task379.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task379.json`
- ARC-GEN task id: `ecdecbb3`
- ARC-DSL task id: `ecdecbb3`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_ecdecbb3.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_ecdecbb3.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task379.py`

## Pattern

The input is a black grid containing one or two full cyan (`8`) guide lines and several isolated red (`2`) pixels away from those guide lines. In the untransposed form the guide lines are horizontal; some examples are transposed, so the same rule may appear with vertical guide lines.

For each red pixel, project it orthogonally toward each cyan guide line in the same row/column direction. Paint the straight segment red from the original red pixel to the guide line when the path reaches that guide line without crossing a cyan cell that was already painted into the output. At the contact cell on the guide line, paint the 8-neighborhood cyan and then restore the contact center to red. If a candidate path hits cyan before reaching the intended line, that path is blocked and is not painted.

The output has the same size as the input.

## Readable Python Solver

```python
def solve(grid):
    h, w = len(grid), len(grid[0])

    def run_untransposed(g):
        h, w = len(g), len(g[0])
        out = [row[:] for row in g]
        guide_rows = [
            r for r in range(h)
            if all(g[r][c] == 8 for c in range(w))
        ]
        reds = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if g[r][c] == 2
        ]
        for r0, c in reds:
            for line in guide_rows:
                step = -1 if line < r0 else 1
                r = r0
                while r != line:
                    if out[r][c] == 8:
                        break
                    out[r][c] = 2
                    r += step
                if r == line:
                    for dr in (-1, 0, 1):
                        rr = r + dr
                        if not 0 <= rr < h:
                            continue
                        for dc in (-1, 0, 1):
                            cc = c + dc
                            if 0 <= cc < w:
                                out[rr][cc] = 8
                    out[r][c] = 2
        return out

    row_lines = [r for r in range(h) if all(grid[r][c] == 8 for c in range(w))]
    if row_lines:
        return run_untransposed(grid)

    transposed = [list(row) for row in zip(*grid)]
    solved = run_untransposed(transposed)
    return [list(row) for row in zip(*solved)]
```

## Generator Constraints

- Width and height are independently sampled from 12 through 20.
- The generator uses only black (`0`), red (`2`), and cyan (`8`).
- There is either one guide line at `height // 2` or two guide lines in the upper and lower middle bands.
- Red pixels are sampled in interior columns only (`1..width-2`) so the 3x3 cyan halo around a contact point stays in bounds horizontally.
- Red pixels are never placed on a guide line or immediately adjacent to one.
- Red columns are spaced to avoid overlapping boxes for many cases; with two guide lines, closer columns can coexist when the red pixels lie in separated bands around the two guide lines.
- A transposition branch swaps rows and columns after generating the horizontal-line version, so optimizers must handle both orientations.
- The generator retries until at least one output cell differs from input.

## Reference Notes

ARC-GEN is the most explicit source for sequential blocking: the output buffer is checked while painting each red path, so a previously created cyan halo can block a later path in the same column. ARC-DSL expresses the same idea as red/cyan object pairs, `gravitate`, connecting the red center toward the cyan object, keeping only long enough connections, and filling neighbors of the contact point cyan. The Code Golf solution confirms a compact regex-based orientation trick but is too compressed to use as the primary readable rule.

There is no disagreement between the references. The main ambiguity for graph design is not the rule itself but how to represent both orientations and the cyan-halo blocking cheaply.
