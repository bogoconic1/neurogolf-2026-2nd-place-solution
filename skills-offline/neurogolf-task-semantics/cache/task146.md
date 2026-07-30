# task146 Semantics

## Sources

- Current champion builder: `solutions_py/task146.py`
- Bundled task examples: `evaluate/scripts/neurogolf-2026/task146.json`
- ARC-GEN task id: `662c240a`
- ARC-DSL task id: `662c240a`
- ARC-GEN generator: `reference/scripts/ARC-GEN/tasks/task_662c240a.py`
- ARC-DSL solver: `reference/scripts/arc-dsl/solver_scripts/solve_662c240a.py`
- Code Golf 2025 solution: `reference/scripts/NeurIPS-Code-Golf-2025/solutions/task146.py`

## Pattern

The input is a 9x3 grid made of three stacked 3x3 blocks. Exactly two blocks are symmetric across the main diagonal, and exactly one block is asymmetric across the main diagonal. The output is the asymmetric 3x3 block unchanged.

Each 3x3 block uses two colors in generated examples. The six colors used across the three blocks are all present at least once in the full input. Symmetric blocks are forced by mirroring the lower triangle into the upper triangle. The asymmetric block is the only one whose cell `(r,c)` differs from `(c,r)` for at least one off-diagonal pair.

## Readable Python Solver

```python
def solve(grid):
    for start in (0, 3, 6):
        block = [row[:] for row in grid[start:start + 3]]
        symmetric = all(block[r][c] == block[c][r] for r in range(3) for c in range(3))
        if not symmetric:
            return block
    raise ValueError("no asymmetric block")
```

## Generator Constraints

- Input shape is always 9 rows by 3 columns.
- Output shape is always 3 rows by 3 columns.
- There are three candidate 3x3 blocks at row offsets 0, 3, and 6.
- The asymmetric block index `idx` is sampled from 0, 1, or 2.
- Six random colors are sampled; block `i` uses only the two colors assigned to that block.
- The generator rejects cases where fewer than six colors appear, so every assigned color appears at least once.
- For the two non-target blocks, lower-triangle cells are mirrored across the main diagonal to make the block symmetric.
- For the target block, generation repeats until at least one off-diagonal pair remains unequal.

## Reference Notes

ARC-DSL splits the input vertically into three 3x3 grids, checks whether `dmirror(block) == block`, flips that equality, and extracts the first block for which the flipped predicate is true. Code Golf uses the same recursive idea: test the top 3x3 block against its transpose, return it if asymmetric, otherwise recurse on the remaining rows. ARC-GEN confirms there is exactly one asymmetric block.

There is no ambiguity: preserve the chosen 3x3 block exactly, including its colors and geometry.
