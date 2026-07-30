---
name: neurogolf-runtime-v1
description: >-
  Actively DRIVE the NeuroGolf runtime-v1 sweep. Use when asked to run, resume,
  create sessions for, harvest, submit, or reconcile the runtime-focused
  Claude-in-Chrome NeuroGolf sweep. This lane optimizes hosted/local runtime,
  strictly by reordering operands inside existing high-arity Einsum nodes while
  preserving baseline behavior, no fresh pass-rate regression, and the exact
  same score, using the empty
  claude-for-chrome-runtime-v1 artifact folders, runtime-v1 manifests, a 3700s
  submit timeout, and runtime-specific submitted_by labels.
---

# NeuroGolf runtime-v1 — active runtime sweep driver

This skill drives the runtime-v1 sweep.

Local artifact roots:

- ONNX: `claude-for-chrome-runtime-v1/{safe,unsafe,invalid,timeout}`
- Markdown: `claude-for-chrome-runtime-v1-markdown/`
- Baseline builds for analysis only: `zzz_sessions/runtime-v1-baselines/`

State/manifests:

- Task list: `zzz_sessions/neurogolf_runtime_v1_tasklist.txt`
- Session map: `zzz_sessions/neurogolf_runtime_v1_session_task_mapping.md`
- ONNX manifest: `zzz_sessions/onnx-harvest-manifest-runtime-v1.csv`
- Markdown manifest: `zzz_sessions/markdown-harvest-manifest-runtime-v1.csv`
- Blockers: `zzz_sessions/runtime-v1-blockers.md`

## Runtime Objective

Primary success is lower runtime with baseline behavior and the **same score**
preserved. This lane is not for score golf.

Correctness is measured relative to the source/baseline graph, not against an
ideal perfect solver. The reordered graph should do the same thing as the
baseline. If the baseline passes `p%` of fresh samples, the reordered graph must
pass at least `p%` on a comparable fresh set. Prefer direct baseline-output
parity when practical; at minimum, do not reduce the baseline's fresh pass rate
or hosted/static score.

The only allowed transformation is reordering operands of existing `Einsum`
nodes. Do not introduce graph staging, operator replacements, score-neutral
simplifications, shape rewrites, extra helper nodes, initializer changes, or any
semantic rewrite outside `Einsum` operand order.

Prefer tasks with high `base_eval`, high local ORT time, or large
`Einsum` trace proxies (`sum_live`, peak, `steps>=1m`, `steps>=5m`).
During operand-order search, minimize `sum_live` as the cheap primary proxy.
Use lower `peak` and fewer high-live steps as tie-breakers and spike checks.
Do not accept a candidate solely because the proxy improves; validate behavior
and actual ORT/hosted runtime before keeping it.

Use `skills-offline/einsum-runtime` only as analysis. Do not treat it as an
optimizer. The agent must design and validate its own graph.

If ONNX Runtime is too slow for iteration, prototype candidate operand orders
with NumPy `einsum` on extracted or representative tensors. NumPy timing is an
analysis shortcut only; the final artifact must still be an ONNX graph with only
existing `Einsum` operand order changed, and it must still validate under the
NeuroGolf evaluator/ORT path before saving.

## Hard Rules

- **MODEL MUST BE `Extra High`.** Before creating ANY session, open the composer model picker (Chrome tool)
  and confirm `Extra High` is selected; select it if not. Do NOT use Pro Extended or any other tier — `Extra High` ONLY.
  Re-confirm whenever the picker may have reset.
- If starting from `skills-offline/solutions_py/taskNNN.py`, build a persistent
  ONNX before tracing or timing:
  `/Users/geremieyeo/anaconda3/bin/python skills-offline/solutions_py/taskNNN.py zzz_sessions/runtime-v1-baselines/taskNNN_baseline.onnx`.
  Baseline builds in `zzz_sessions/runtime-v1-baselines/` are analysis inputs,
  not submission artifacts.
- Only `Einsum` operand reordering is allowed. The candidate must be
  behaviorally equivalent to the source graph and must keep the exact same score.
  If the baseline is imperfect on fresh samples, the candidate must be no worse
  than the baseline fresh pass rate.
- NumPy `einsum` may be used to benchmark operand orders when ORT iteration is
  too slow, but NumPy code is never the submitted artifact.
- Use a **3700 second submit timeout** for runtime submissions.
- Send **1 message per task: Round 1 only.** Round 2 is a **harvest-only** pass —
  collect the candidate ONNX + markdown the model produced after Round 1 and submit
  them, but send **NO** further message. So each task gets 1 message across 2 passes
  (send, then harvest). Track the sent count as `N/1` in the session mapping `Rounds`
  column. Once a task reaches `1/1`, do the Round-2 harvest, mark it complete, and skip
  it unless the user explicitly re-enables follow-up nudges.
- Harvest markdown every round. Harvest ONNX candidates when they are worth
  testing for same-score runtime improvement from `Einsum` operand reordering.
  Also harvest any candidate that improves score versus the baseline, even if it
  is outside the runtime-only target; mark/report it as a better-score exception.
- Classify ONNX by the session's validation at harvest:
  - `safe`: baseline-parity or fresh-no-regression validated with no known
    score regression.
  - `unsafe`: fresh failures, incomplete validation, risky generator assumption,
    baseline probe, or any candidate that may still be useful for runtime study.
  - `invalid`/`timeout` are assigned by the submitter/reconciler from the Space
    verdict, not by the browser harvester.
- Never block the round-robin on one bad session. Log blockers and move on.
- `~/Downloads` is shared. Always `snap` before clicking download cards and
  `move` immediately after.
- **2-min cooldown = ≥120 s between message SENDS, and the harvest counts toward it.** There
  is ONE place you wait: right before sending a task's message, *after* you've clicked that
  task's ONNX/markdown downloads. At that point, check how long since the **previous** send —
  the time you just spent opening the session and downloading counts — and wait only
  `max(0, 120 − seconds_since_last_send)` (background sleep). In round 2+ that is usually **0**
  (the harvest already took >120 s → send immediately). Round-1 creates have no harvest before
  the send, so consecutive creates wait ~the full 120 s. **Never blind-sleep a fixed 120/125 s
  after every send** — that double-counts the harvest you just did. Stamp each send's wall-clock
  time (`Last sent`) so the next gap can be computed.

## Setup

Create missing local infrastructure:

```bash
mkdir -p claude-for-chrome-runtime-v1/{safe,unsafe,invalid,timeout}
mkdir -p claude-for-chrome-runtime-v1-markdown zzz_sessions/runtime-v1-baselines
```

Start pollers from the repo root. These drain files produced by browser harvests:

```bash
PY=/Users/geremieyeo/anaconda3/envs/qgentic-ai/bin/python

$PY auto_post_md.py \
  --watch-dir claude-for-chrome-runtime-v1-markdown \
  --state-file auto_post_md_state_claude_for_chrome_runtime_v1.json \
  --repo golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/runtime-v1-post.log 2>&1 &

$PY auto_submit_onnx.py \
  --watch-dir claude-for-chrome-runtime-v1/safe \
  --submitted-by claude-for-chrome-runtime-v1 \
  --max-age-seconds 0 \
  --submit-timeout 3700 \
  --state-file auto_submit_onnx_state_claude_for_chrome_runtime_v1.json \
  --space golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/runtime-v1-submit-safe.log 2>&1 &

$PY auto_submit_onnx.py \
  --watch-dir claude-for-chrome-runtime-v1/unsafe \
  --submitted-by claude-for-chrome-runtime-v1-fresh-unsafe \
  --max-age-seconds 0 \
  --submit-timeout 3700 \
  --state-file auto_submit_onnx_state_claude_for_chrome_runtime_v1.json \
  --space golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/runtime-v1-submit-unsafe.log 2>&1 &
```

**Start all three pollers (1 markdown poster + 2 ONNX submitters) BEFORE any browser action** — they
drain asynchronously while you drive. Launch each as a plain `… &`; do NOT combine `nohup … &` with a
backgrounded harness call (launch as plain `&` and poll the logs). Confirm each log prints its
`Watching ...` startup line before browser work.

## Browser Workflow

Use the claude-in-chrome MCP/browser tool for browser actions.

For each task in `neurogolf_runtime_v1_tasklist.txt`:

1. **Select the right browser/account first:** `list_connected_browsers` → confirm the correct Chrome
   (the runtime project the user specified) → `select_browser`/`switch_browser` → reuse the project tab.
   Then open or create the task's runtime-v1 session there.
2. If the only local source is `skills-offline/solutions_py/taskNNN.py`, build
   `zzz_sessions/runtime-v1-baselines/taskNNN_baseline.onnx` before asking the
   browser agent to trace or time the graph.
3. Harvest the previous completed round before sending the next message.
4. Download all markdown cards for the task.
5. Download ONNX for candidates that materially help runtime study through
   same-score `Einsum` operand reordering, explicit user-requested comparison,
   or any better-score result.
6. On **Round 1 only**, send the Round-1 prompt and increment `Rounds` to `1/1`;
   on **Round 2** send no message — just harvest (steps 3–5) and mark complete.
   Then move to the next task under the normal send-to-send cooldown.

**Browser gotchas (apply every round):**
- **Before sending, open the "Pasted Text" chip** to confirm the right `taskNNN` is inside — the shared
  macOS clipboard can be clobbered by a parallel driver. After sending, **verify the user bubble appeared**.
- The round-1 prompt is a paste — make sure the **FULL text landed** in the composer and the user message
  **actually posted** before moving on; if not, re-open the composer, ensure the full text is present, send
  again, verify the bubble.
- **Markdown harvest:** ChatGPT lazy-loads older turns **only on a real wheel scroll** — scroll to the very
  TOP first, then download each card top→bottom (chronological), then run the md mover.
- **READ `GPT56_BUTTON_PATTERNS.md` (repo root) BEFORE harvesting — it is the guide for clicking the ONNX
  download buttons.** GPT-5.6/5.5 renders the download control in several distinct shapes (prefixed link,
  bare chip, champion-skip, descriptive) plus INERT non-downloadable chips; the doc shows how to recognize
  and click each shape and which to skip. **ONNX cards download reliably only via a coordinate click**, not
  ref/JS clicks.
- **Best ONNX not downloadable** (no card / inert link / `/tmp`-only path) and worth grabbing → do ONE
  recovery probe: send `Give me the downloadable best ONNX file`, wait ~2 min (the probe is a send, so that
  wait is the cooldown), then download it or add a `runtime-v1-blockers.md` row. Don't loop.
- **ZIP fallback:** ONNX/markdown delivered only inside a zip → download it, `unzip -o -j '*.onnx'` (or
  `'*.md'`), move each into its folder as `task{NNN}_<name>.onnx`/`.md`, append a manifest row.
  **Generic-name markdown** ("no md matching taskNNN") → inspect content for the task number and `mv` it in.
- **Checkpoint** the mapping (and the driver memory, if present) every ~10 sends; the manifests + mapping
  are the source of truth, and both pollers content-hash dedup, so re-harvest/re-submit/re-post is idempotent.

Sweep **2 passes** over the task list: **Round 1** sends the prompt to each task;
**Round 2** harvests each task's output (candidate ONNX + markdown) and sends **no**
message. Do not send a second message to any task unless the user explicitly
re-enables follow-up nudges.

If the runtime project URL/session mapping is missing, create sessions in the
runtime project URL the user provides and record the new `/c/<session_id>` URL in
the mapping file.

## Mover Commands

Run from repo root.

```bash
PY=/Users/geremieyeo/anaconda3/bin/python

# ONNX: snap before clicking, move after. Labels: safe | unsafe | skip
$PY .claude/skills/neurogolf-runtime-v1/scripts/harvest_move_runtime_v1.py snap
$PY .claude/skills/neurogolf-runtime-v1/scripts/harvest_move_runtime_v1.py move <task> <sid> <label...>

# Markdown: snap before clicking, move after.
$PY .claude/skills/neurogolf-runtime-v1/scripts/harvest_move_md_runtime_v1.py snap
$PY .claude/skills/neurogolf-runtime-v1/scripts/harvest_move_md_runtime_v1.py move <task> <sid>
```

If the ONNX mover reports a label/file mismatch, stop and recount downloads. Do
not force it.

## Runtime Prompt Templates

Round 1 prompt:

```text
Optimize task{NNN} for NeuroGolf runtime by Einsum operand reordering only. Use the provided task{NNN}.py Python builder as the source of truth; do not assume a prebuilt ONNX file. First read skills-offline/einsum-runtime/SKILL.md if it is available. Build a baseline ONNX, for example `python task{NNN}.py task{NNN}_baseline.onnx`, or `python skills-offline/solutions_py/task{NNN}.py task{NNN}_baseline.onnx` if that repo layout exists, then trace that ONNX. Measure baseline behavior, score, fresh pass rate when available, and runtime, then run the einsum_runtime_trace.py script for every large Einsum. The target graph change is to reorder operands inside existing Einsum nodes. The candidate should do the same thing as the baseline and should keep the exact same score, memory, params, and operator set except for Einsum input order/equation operand order. If the baseline passes p% of fresh samples, the candidate must pass >= p% on a comparable fresh set. Do not add staging, replacements, simplifications, helper nodes, initializer changes, shape rewrites, or any non-Einsum semantic change for the runtime lane. sum_live approximates total intermediate tensor traffic during left-to-right Einsum execution. It sums the live intermediate size after each operand is incorporated and all currently-dead axes are reduced. Lower is generally better for runtime. Reordering the Einsum operands alone can cut sum_live by orders of magnitude (e.g. 100M -> 500K) with the EXACT same output, so search hard over operand orders. Actively minimize sum_live as the cheap primary search proxy; use peak and steps>=1m/steps>=5m as tie-breakers and spike checks. If ORT is too slow for iteration, use NumPy einsum to benchmark candidate operand orders on extracted or representative tensors, then implement only the best operand order back into ONNX. Validate the final ONNX against the baseline and with evaluate/scripts/evaluate.py plus ORT timing before saving. Save any useful ONNX candidate and a markdown report with before/after score, fresh pass rate or failure count if measured, memory, params, hosted runtime if available, NumPy timing if used, and trace metrics. If you accidentally find a better-score candidate, save it and report it clearly as a better-score exception; do not discard it just because runtime-v1 targets same-score reorders. Do not copy a hidden fast graph unless explicitly provided.
```

> **NOTE:** The three follow-up nudges below (Round 2+ default / Coverage / Scorer) are
> **NOT sent under the current 1-round policy** — Round 2 only harvests. They are kept for
> reference in case follow-up nudges are re-enabled.

Round 2+ default nudge:

```text
Continue optimizing runtime by Einsum operand reordering only. If you need the baseline ONNX again, rebuild it from the provided task Python builder. Re-check the slowest existing Einsum, run the runtime trace again, and try a materially different operand order that lowers sum_live (reordering alone can cut sum_live by orders of magnitude, e.g. 100M -> 500K, with the EXACT same output), using peak and high-live steps as tie-breakers and spike checks. If ORT is too slow for iteration, benchmark candidate operand orders with NumPy einsum, then validate the final ONNX under ORT/evaluate. Preserve baseline behavior, no fresh pass-rate regression, and the exact same score for runtime-lane candidates. Do not add staging, replacements, helper nodes, initializer changes, shape rewrites, or any non-Einsum semantic change for runtime-lane candidates. Report before/after sum_live, peak, high-live steps, score, fresh pass rate or failure count if measured, params, memory, NumPy timing if used, and actual ORT runtime where possible. If a candidate improves score versus the baseline, save it and report it as a better-score exception even if it is not a pure same-score runtime reorder.
```

Coverage nudge:

```text
Your last candidate regressed relative to the baseline. Fix baseline behavior or fresh pass-rate parity first, then re-measure runtime. Do not keep a faster graph unless it preserves the same score and is no worse than the baseline on bundled and fresh examples.
```

Scorer nudge:

```text
Your last ONNX was rejected or unstable under the scorer/onnxruntime path. Make it load, infer, and run under the evaluator first; then resume runtime optimization.
```

## Reconcile

After pollers have submitted files, reconcile by Space verdict:

```bash
PY=/Users/geremieyeo/anaconda3/bin/python
$PY .claude/skills/neurogolf-runtime-v1/scripts/reconcile_submissions_runtime_v1.py --dry-run
$PY .claude/skills/neurogolf-runtime-v1/scripts/reconcile_submissions_runtime_v1.py --apply
```

Expected routing:

- `ok`: leave where it is.
- `unsafe` or `wrong`: move to `unsafe/`.
- `error` or `eval_error`: move to `invalid/`.
- `timeout`: move to `timeout/`.

Note: an `ok` in `unsafe/` stays unsafe (no promotion); a safe-labeled file the Space marks `unsafe`/`error`
MOVES (the Space caught what the session missed — dedup may skip re-posting under the unsafe author, which
the reconciler flags).

## Verification

```bash
PY=/Users/geremieyeo/anaconda3/envs/qgentic-ai/bin/python
# pollers alive
ps aux | grep -E 'auto_post_md|auto_submit_onnx' | grep -v grep | grep runtime-v1
# folder counts
for d in safe unsafe invalid timeout; do printf "%s " $d; ls claude-for-chrome-runtime-v1/$d/*.onnx 2>/dev/null | wc -l; done
ls claude-for-chrome-runtime-v1-markdown/*.md 2>/dev/null | wc -l
# posted so far (content-unique)
$PY -c "import json;print(len(json.load(open('auto_post_md_state_claude_for_chrome_runtime_v1.json')).get('done',[])))"
```
