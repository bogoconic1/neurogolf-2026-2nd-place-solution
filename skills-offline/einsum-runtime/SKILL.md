---
name: einsum-runtime
description: Trace high-arity ONNX Einsum operand order and estimate runtime risk using live-axis proxies such as peak live size, cumulative live size, and large-live-step counts. Use as analysis-only guidance for same-score NeuroGolf Einsum operand reordering, including NumPy einsum timing triage when ONNX Runtime is too slow.
---

# Einsum Runtime Skill

Use this skill to analyze high-arity NeuroGolf `Einsum` graphs and quantify why
one operand order may be expensive. This skill is analysis-only: it reports
runtime-risk proxies and trace rows, but it does not rewrite graphs or choose an
optimization.

The trace script reads ONNX, not Python builders. If the source is
`skills-offline/solutions_py/taskNNN.py`, build the ONNX first:

```bash
mkdir -p zzz_sessions/runtime-v1-baselines
/Users/geremieyeo/anaconda3/bin/python \
  skills-offline/solutions_py/taskNNN.py \
  zzz_sessions/runtime-v1-baselines/taskNNN_baseline.onnx
```

For runtime-v1 work, the only allowed downstream graph change is reordering
operands inside existing `Einsum` nodes. Do not use this skill to justify graph
staging, replacing `Einsum` with other ops, adding helper nodes, changing
initializers, changing shapes, or changing semantics. Candidate ONNX files must
keep the same score as the baseline and must not regress baseline behavior: if
the baseline passes `p%` of fresh samples, the reordered model should pass
`>= p%` on a comparable fresh set.

If an agent discovers a better-score candidate while doing runtime analysis,
harvest and report it as a better-score exception. Do not discard it solely
because this skill's normal target is same-score runtime analysis.

The script is not an ONNX Runtime profiler. It parses every `Einsum` in the
model, uses ONNX shape inference to resolve operand shapes, and simulates the
explicit left-to-right operand order. At each operand it:

1. Adds the operand's labels to the live label set.
2. Removes reduction labels whose final occurrence has just been consumed.
3. Records the product of the remaining live label dimensions.

This gives three useful warning signals:

- `peak`: largest live-shape proxy seen at any step.
- `sum_live`: cumulative live-shape proxy across all operand steps.
- `steps>=...`: number of steps whose live-shape proxy crosses a threshold.

Use actual CPU/GPU `evaluate` timing as the final authority. These proxies are
for triage only.

If ONNX Runtime is too slow for inner-loop experimentation, use NumPy `einsum`
to benchmark candidate operand orders on extracted or representative tensors.
Treat NumPy timing as triage only: the final artifact still has to be ONNX and
must be validated against the baseline under the evaluator/ORT path.

## Agent Workflow

When analyzing a slow `Einsum` graph:

1. Run the trace on the slow graph and record `peak`, `sum_live`, and
   large-live-step counts.
2. Inspect operand rows around the largest `live` values.
3. Identify which labels remain in `alive` for many rows and which later
   operands finally reduce them.
4. If a faster comparison graph exists, run the same trace on both graphs and
   compare summaries plus the row ranges where the slow graph carries extra
   labels.
5. For runtime-v1, use the trace to propose `Einsum` operand orders only. Keep
   the final graph structure unchanged except for `Einsum` input/equation
   operand order.
6. Stop at analysis. Do not use this skill to rewrite the ONNX. The optimizing
   agent should use these measurements to design its own candidate and validate
   same score, baseline no-regression, and actual runtime.

The script intentionally reads the input order from the ONNX node. ONNX
Runtime's CPU `Einsum` implementation processes operands pairwise in that order,
so the trace is useful evidence when an agent is reasoning about runtime.

## Setup

Set `SKILL_DIR` to this folder. Dependencies are usually already available in
the local NeuroGolf environment. If import fails, install:

```bash
python -m pip install -r "$SKILL_DIR/requirements.txt"
```

## Trace One Model

Build from `solutions_py` first when needed:

```bash
mkdir -p zzz_sessions/runtime-v1-baselines
python skills-offline/solutions_py/task369.py zzz_sessions/runtime-v1-baselines/task369_baseline.onnx
```

```bash
python "$SKILL_DIR/scripts/einsum_runtime_trace.py" /path/to/taskXXX.onnx
```

For a compact `sum_live` profile:

```bash
python "$SKILL_DIR/scripts/einsum_runtime_trace.py" /path/to/taskXXX.onnx --summary-only
```

The output contains the equation, inferred label dimensions, summary metrics,
and one row per operand:

```text
00 input  nawb   live=     9000 reduce=         alive=abnw
01 F      fe     live=   810000 reduce=         alive=abefnw
02 F      ie     live=  2430000 reduce=         alive=abefinw
03 F      pe     live=   243000 reduce=e        alive=abfinpw
```

`reduce` lists labels removed at that step. `alive` lists labels still carried
into later pairwise contractions.

## Options

```bash
# Only summary lines
python "$SKILL_DIR/scripts/einsum_runtime_trace.py" model.onnx --summary-only

# Print only the first 20 operands
python "$SKILL_DIR/scripts/einsum_runtime_trace.py" model.onnx --limit 20
```

## Interpretation

`sum_live` is the sum of every per-operand `live` value. High `sum_live` means
the graph carries large intermediate label sets across many pairwise steps. It
often tracks sustained CPU work better than `peak` alone, but it is still only a
proxy. Use it to explain and prioritize investigation, not as a replacement for
actual ONNX Runtime timing.

The per-row fields are:

- `live`: product of dimensions for labels still alive after this operand.
- `reduce`: labels whose final non-output occurrence ended at this operand.
- `alive`: labels still carried into later operands.

## Runtime-v1 Report Checklist

When using this skill for the runtime-v1 sweep, report:

- source/baseline ONNX path
- before/after score, memory, params, and operator set; these should match
- before/after fresh pass rate or failure count if measured
- before/after ORT timing, plus NumPy `einsum` timing if used for triage
- before/after `sum_live`, `peak`, and large-live-step counts
- exact `Einsum` operand order changes
- any better-score exception found during analysis
