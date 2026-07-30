---
name: solver-search
description: Use Z3 SMT or OR-Tools CP-SAT for bounded algebraic expression search, integer/boolean formula synthesis, equivalence checking, compact constant selection, and NeuroGolf micro-golf graph rewrites.
---

# Solver Search Skill

Use this skill when a task needs exact bounded search instead of manual guessing: small arithmetic formulas, boolean masks, selector logic, compact ONNX constants, QLinearConv encodings, or proof that two rewrites are equivalent on a finite domain.

Keep solver code as local tooling. Do not import `z3` or `ortools` from a submitted artifact unless that target environment explicitly allows those dependencies.

## Setup

Set `SKILL_DIR` to this skill folder, then install dependencies:

```bash
python -m pip install -r "$SKILL_DIR/requirements.txt"
```

Run the smoke test:

```bash
python "$SKILL_DIR/scripts/smoke_test.py"
```

The smoke test exercises both backends and prints JSON with one Z3 polynomial-equality encoding and one CP-SAT linear-threshold synthesis.

## Choosing a Backend

- Use **Z3** for bit-vector or integer semantics, nonlinear-looking expressions over constants, boolean equivalence checks, and proving that a rewrite matches every value in a small bounded domain.
- Use **CP-SAT** for discrete choices, linear integer constraints, operator/constant selection, and lexicographic cost minimization.
- Use plain brute force when the full search has fewer than a few million candidates; it is often simpler and faster than a solver.

## Workflow

1. Extract a small, explicit input/output table from train, test, and any trustworthy generated examples.
2. Bound every variable tightly: grid coordinates, colors `0..9`, labels, kernel weights, scales, or small constants.
3. Encode correctness first, then add an objective that approximates the target cost: fewer params, fewer intermediates, smaller dtype, or fewer ONNX nodes.
4. Ask the solver for one candidate, translate it into the target implementation, and validate with the target evaluator.
5. If the candidate fails outside the search table, add the counterexample and re-run.

## NeuroGolf Patterns

- Synthesize shifted-label predicates such as `w1*x + w2*x*x + b > 0` for `x == color+1` while keeping padded sentinel labels non-positive.
- Search tiny `Conv`/`QLinearConv` kernels, route weights, and biases for color encodings before writing them by hand.
- Search row/column masks as linear thresholds over `r`, `c`, counts, or object coordinates.
- Check whether a scalar-label tail, one-hot tail, `Pad+Equal`, or `Equal+Pad` rewrite is equivalent over all colors and active cells.

Solver output is only a hypothesis. Always run the evaluate/metric checks after translating it into ONNX, because solver cost is only a proxy for the hosted scorer.
