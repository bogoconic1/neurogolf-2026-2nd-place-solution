# task191 Semantics

## Sources

- Current champion builder: `solutions_py/task191.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task191.json`
- ARC-GEN task id: `7df24a62`
- ARC-DSL task id: `7df24a62`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_7df24a62.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_7df24a62.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task191.py`

## Pattern

The input is a fixed 23x23 grid using black (`0`), blue (`1`), and yellow (`4`). One yellow sprite pattern is shown inside a solid blue reference frame. The frame is one cell larger than the sprite on all four sides. Elsewhere in the grid are yellow dots: some are random noise and some form copies of the reference yellow pattern under rotation and/or reflection. The output preserves every yellow dot and draws the same one-cell-thick blue frame around every detected copy of the yellow pattern. The reference frame remains blue as in the input.

## Readable Python Solver

```python
def solve(grid):
    H, W = len(grid), len(grid[0])
    blue = [(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 1]
    r0, r1 = min(r for r, _ in blue), max(r for r, _ in blue)
    c0, c1 = min(c for _, c in blue), max(c for _, c in blue)

    # Yellow pattern inside the blue reference frame.
    pattern = []
    for r in range(r0 + 1, r1):
        row = []
        for c in range(c0 + 1, c1):
            row.append(1 if grid[r][c] == 4 else 0)
        pattern.append(row)

    def rot90(p):
        return [list(row) for row in zip(*p[::-1])]

    def transforms(p):
        out = []
        q = p
        for _ in range(4):
            out.append(q)
            out.append([row[::-1] for row in q])
            q = rot90(q)
        # De-duplicate symmetric patterns.
        seen, uniq = set(), []
        for t in out:
            key = tuple(tuple(row) for row in t)
            if key not in seen:
                seen.add(key)
                uniq.append(t)
        return uniq

    yellow = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 4}
    out = [row[:] for row in grid]

    for pat in transforms(pattern):
        th, tw = len(pat), len(pat[0])
        for top in range(H - th + 1):
            for left in range(W - tw + 1):
                ok = True
                for dr in range(th):
                    for dc in range(tw):
                        want = pat[dr][dc] == 1
                        have = (top + dr, left + dc) in yellow
                        if want != have:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                # The generator rejects cases where yellow would lie on the frame.
                for r in range(top - 1, top + th + 1):
                    for c in range(left - 1, left + tw + 1):
                        if 0 <= r < H and 0 <= c < W:
                            out[r][c] = 1

    for r, c in yellow:
        out[r][c] = 4
    return out
```

## Generator Constraints

ARC-GEN uses `size=23`. The reference sprite has `tall` in `1..3` and `wide` in `2..3`; the sampled pattern has `max(wide, tall)` yellow cells and is constrained to touch the top, bottom, left, and right edges of its own sprite rectangle. There are `2..6` sprite placements. The last placement is the unrotated reference and is placed at least one cell away from the grid edge so its blue frame fits. Earlier placements may be rotated by 0, 90, 180, or 270 degrees; the ARC-DSL solver also considers transposed/reflected variants when matching. Random yellow noise is added at about 5 percent density, but placement generation clears candidate frame areas and retries until blue frames do not collide and yellow dots do not fall on a frame border. Inputs and outputs are always 23x23.

## Reference Notes

The ARC-DSL solver extracts the blue reference subgrid, forms its four rotations, also accounts for transposed/reflected orientations, then scans possible placements. For each orientation it compares yellow cells after treating blue as black, finds exact zero-sized differences, and fills blue frames around every match. The Code Golf solution performs the same repeated rotation/reflection search with regular-expression style matching. The references agree with ARC-GEN.
