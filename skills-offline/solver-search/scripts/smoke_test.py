#!/usr/bin/env python3
"""Smoke-test Z3 and OR-Tools CP-SAT for tiny synthesis tasks."""

from __future__ import annotations

import json


def run_z3_shifted_label() -> dict:
    from z3 import Int, Solver, sat

    target = 4
    w1, w2, b = Int("w1"), Int("w2"), Int("b")
    s = Solver()
    for v in (w1, w2, b):
        s.add(v >= -20, v <= 20)
    for x in range(11):
        expr = w1 * x + w2 * x * x + b
        s.add(expr >= 1 if x == target else expr <= 0)
    assert s.check() == sat
    m = s.model()
    coeffs = {name: m[var].as_long() for name, var in [("w1", w1), ("w2", w2), ("b", b)]}
    values = [coeffs["w1"] * x + coeffs["w2"] * x * x + coeffs["b"] for x in range(11)]
    return {
        "target": target,
        "coeffs": coeffs,
        "positive_at": [i for i, value in enumerate(values) if value > 0],
        "verified": [i for i, value in enumerate(values) if value > 0] == [target],
    }


def run_cp_sat_threshold() -> dict:
    from ortools.sat.python import cp_model

    model = cp_model.CpModel()
    a = model.NewIntVar(-4, 4, "a")
    b = model.NewIntVar(-4, 4, "b")
    d = model.NewIntVar(-4, 4, "d")
    for r in range(3):
        for c in range(3):
            expr = a * r + b * c + d
            if r >= c:
                model.Add(expr >= 1)
            else:
                model.Add(expr <= 0)
    abs_vars = []
    for var, name in [(a, "abs_a"), (b, "abs_b"), (d, "abs_d")]:
        av = model.NewIntVar(0, 4, name)
        model.AddAbsEquality(av, var)
        abs_vars.append(av)
    model.Minimize(sum(abs_vars))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT failed with status {status}")
    coeffs = {"a": solver.Value(a), "b": solver.Value(b), "d": solver.Value(d)}
    positives = [(r, c) for r in range(3) for c in range(3) if coeffs["a"] * r + coeffs["b"] * c + coeffs["d"] > 0]
    expected = [(r, c) for r in range(3) for c in range(3) if r >= c]
    return {"target": "r >= c", "coeffs": coeffs, "positive_at": positives, "verified": positives == expected}


def main() -> None:
    result = {"z3_shifted_label": run_z3_shifted_label(), "cp_sat_threshold": run_cp_sat_threshold()}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
