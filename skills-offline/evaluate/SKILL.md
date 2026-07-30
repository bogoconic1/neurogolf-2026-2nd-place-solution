---
name: evaluate
description: Evaluate NeuroGolf submission zips, single ONNX files, or single Python builder files locally against bundled task data, optional fresh ARC-GEN samples, and the leaderboard scoring metric.
---

# Evaluate skill

Use this skill when an LLM agent needs to locally validate a NeuroGolf `submission.zip`, one-task ONNX candidate, or one-task Python builder.

NeuroGolf is an ONNX code-golf competition for solving 400 ARC-style grid transformation tasks with the smallest/cheapest correct computational graphs. A local evaluation checks exact correctness on known task examples, optionally tries fresh ARC-GEN samples for debugging, and computes the same memory/parameter-based score. A task passes when it evaluates to `status=ok`; extra fresh samples are an optional robustness check.

## What is included

The `scripts/` folder is self-contained:

- `evaluate.py`: local evaluator CLI with the same scoring/validation core as the leaderboard server plus direct single-file input conveniences.
- `mapping.csv`: NeuroGolf task ID to ARC-GEN task ID mapping.
- `neurogolf-2026/`: all 400 task JSON files and `neurogolf_utils.py`.
- `ARC-GEN/`: generator subset needed for fresh ARC-GEN samples.

## Setup

Set `SKILL_DIR` to this skill folder, then run commands from any directory using the `$SKILL_DIR` paths shown below. Dependencies are usually already installed in the local environment. Only install them when a command below fails with `ModuleNotFoundError` or a similar import error:

```bash
python -m pip install -r "$SKILL_DIR/requirements.txt"
```

## Inspect task

Use this before designing a solver to understand train/test/arc-gen examples. The task JSON for all 400 tasks is bundled under `scripts/neurogolf-2026/`.

```bash
python "$SKILL_DIR/scripts/task_inspect.py" task101
```

The command prints a compact text report to stdout: a header with `task_id`, `path`, `split_counts`, `selected_splits`, `pairs_shown`, then each pair as `## split[index]  HxW -> HxW` followed by `input:` / `output:` blocks with each grid rendered as digits 0–9 separated by spaces. Colors are the ARC palette: `0` is background; `1..9` are distinct colors. The default `--splits train,test` keeps output bounded; switch with `--splits all` or `--splits arc-gen` when needed.

The bundled ARC-GEN split commonly has 250+ pairs, so always cap with `--max-pairs` when including it:

```bash
python "$SKILL_DIR/scripts/task_inspect.py" task101 --splits arc-gen --max-pairs 5
```

## Quick local-development validation

For normal local development, use the bundled known task data only. This is the default: it validates the Kaggle/known examples in the task JSON and does **not** generate fresh ARC-GEN samples. We have found a few rare cases where additional generated samples are wrong, so rely on the Kaggle/known set first. A candidate is considered passing when it reaches `status=ok` on the bundled known examples.

Evaluate one task directly from an ONNX file or a Python builder when you need raw evaluator output:

```bash
python "$SKILL_DIR/scripts/evaluate.py" --input task101.onnx --tasks task101 --verbose
```

`--input` accepts either a `.onnx` file or a `.py` builder; substitute `solution.py` to evaluate a Python builder. For single-file inputs, `--tasks` may be omitted only when the file name or parent directory contains the task id, such as `task101.onnx` or `task101/solution.py`. Python builders follow hosted leaderboard conventions: define `save_model(out_path)`, define `build_model()` returning an ONNX model, or write `best_onnx/taskXXX.onnx` / `taskXXX.onnx`.

## Optional fresh ARC-GEN debugging

Use `--new-samples N` only when you want an extra debugging suggestion beyond the bundled known examples:

```bash
python "$SKILL_DIR/scripts/evaluate.py" --submission submission.zip --tasks task101 --new-samples 10 --seed 611 --verbose
```

Fresh generated samples are not authoritative and are not part of the profiling trace used for points. Treat failures on them as a debugging hint, not as stronger evidence than the Kaggle/known examples. Use `--seed` for reproducible generated samples. Omit `--seed` to draw fresh random samples each run.

## Evaluator memory breakdown

Use `--memory-breakdown path/to/breakdown.csv` when you want the scorer-side tensor memory distribution from the same profiling trace used for `memory_bytes`:

```bash
python "$SKILL_DIR/scripts/evaluate.py" --input task101.onnx --tasks task101 --memory-breakdown task101-memory.csv
```

This does **not** edit organizer-provided `neurogolf_utils.py`; it mirrors the visible `calculate_memory` algorithm in the evaluator wrapper and writes CSV after scoring. Columns are `task_id, name, bytes`. Each task starts with a `name=_total` row carrying the breakdown sum, then per-tensor rows sorted by descending bytes. Tasks without a measurable breakdown are omitted; their status is already in the main `--csv` output.

Use this when you are already running local validation and want an authoritative breakdown tied to that run. Use the `inspect-onnx` skill when you want richer standalone ONNX inspection, params contributors, static-vs-profiled byte comparisons, or a quick champion triage outside a full evaluation run.

## Selected submission evaluation

Evaluate a selected list of tasks from a submission zip:

```bash
python "$SKILL_DIR/scripts/evaluate.py" --submission selected_submission.zip --tasks task001 4 10-15 --csv selected_results.csv
```

Task arguments accept `taskXXX`, bare numbers, ranges like `10-15`, and comma-separated tokens. Use `--task-file path/to/tasks.txt` for longer lists. Omit `--tasks` to evaluate every task present in the zip.

## Output interpretation

The evaluator prints a human-readable summary and can also write per-task CSV with `--csv`. Use `--verbose` to print per-task CSV to stdout before the summary.

Important CSV fields:

| Field | Meaning |
| --- | --- |
| `status` | `ok`, `unsafe`, `wrong`, `missing`, `error`, or `metric_error` |
| `ready` | `True` only when the task fully passes known examples and any requested fresh samples |
| `kaggle_ready` | `True` when known/static examples pass and score metrics are available |
| `unsafe` | `True` when known/static examples pass but requested fresh ARC-GEN samples fail; treat this as an advisory debugging signal |
| `points` | task score, present for `ok` and `unsafe` measured candidates |
| `memory_bytes`, `params` | measured score components |
| `new_samples_pass`, `new_samples_fail` | optional fresh ARC-GEN debugging results |
| `mismatch_sample` | compact JSON for a representative failing sample |
| `error` | compact JSON error details |

A candidate passes when it reaches `status=ok` on the bundled known examples. `status=unsafe` means the known examples passed but requested fresh ARC-GEN samples failed; treat fresh-sample failures as a debugging hint, not a hard reject.

## Additional options

**Input selection**
- `--input PATH` / `--submission PATH` (aliases): submission zip, one `taskXXX.onnx`, or one Python builder. Defaults to `submission.zip` in the current directory.
- `--tasks ...`: restrict zip evaluation, or identify the task for a single `.onnx`/`.py` input when the file name doesn't already contain a task id. Accepts `taskXXX`, bare numbers, ranges like `10-15`, and comma-separated tokens.
- `--task-file PATH`: read task ids from a text file (one per line, or whitespace/comma separated). Useful when the list is long enough to clutter the command line.

**Parallelism**
- `--workers N`: number of parallel task workers. Defaults to `8`. Set to `1` when you want sequential execution for easier debugging (deterministic stdout order, single-process tracebacks); raise it on a bigger box if you're evaluating many tasks.

**Fresh ARC-GEN debugging**
- `--new-samples N`: generate N fresh ARC-GEN samples per task and run them after the static checks. Default `0`. Failures here mark the candidate `unsafe` but do not affect `memory_bytes` or `params`.
- `--seed N`: reproducible fresh-sample generation. Omit for a fresh random draw each run.

**Output**
- `--verbose`: print the per-task CSV (same 19 columns as `--csv`) to stdout before the summary. Convenient for one or a few tasks; for big runs, prefer `--csv FILE` because the `mismatch_sample` and `error` columns can each be multiple KB per failing row.
- `--csv PATH`: write the per-task metrics CSV to a file. Always safe to use; doesn't bloat stdout.
- `--memory-breakdown PATH`: write the scorer-side tensor memory distribution CSV. Columns `task_id, name, bytes`; each task starts with a `name=_total` row.
- `--no-progress`: silence the stderr progress bar. Useful when capturing output programmatically.
