---
name: neurogolf-cost-golf-v1
description: >-
  Actively DRIVE the NeuroGolf cost-golf-v1 sweep. Use when asked to run, resume, create sessions
  for, harvest, submit, or reconcile the general COST Claude-in-Chrome NeuroGolf sweep. This lane
  takes each task's champion — which may be ANY op family (Einsum, ConvTranspose, Conv, TfIdfVectorizer,
  the random-generator ops, memorizers, …) — and lowers its TOTAL cost memory_bytes+params
  (score = 25 − ln(memory+params)) by any construction. The final deliverable may carry memory and need not
  resemble the champion. Uses the COST_GOLF_STARTER.txt playbook,
  claude-for-chrome-cost-golf-v1 artifact folders, cost-golf-v1 manifests, a 3700s submit
  timeout, and cost-golf-v1 submitted_by labels.
---

# NeuroGolf cost-golf-v1 — "beat any champion on total cost" sweep driver

This is the **general** cost lane: it puts no constraint on the shape of the answer. Each task's champion may
be ANY op family, and this lane **lowers its total cost `memory_bytes + params`** using the
`COST_GOLF_STARTER.txt` playbook. The candidate may replace the champion outright and may carry memory.
Score = `25 − ln(memory + params)`, so **+1.0 means cutting total cost to ≈ cost/e** (÷2.718).

Local artifact roots:

- ONNX: `claude-for-chrome-cost-golf-v1/{safe,unsafe,invalid,timeout}`
- Markdown: `claude-for-chrome-cost-golf-v1-markdown/`
- Baseline builds for analysis only: `zzz_sessions/cost-golf-v1-baselines/`

State/manifests:

- Task list: `zzz_sessions/neurogolf_cost_golf_v1_tasklist.txt` (any task whose champion cost looks beatable)
- Session map: `zzz_sessions/neurogolf_cost_golf_v1_session_task_mapping.md`
- ONNX manifest: `zzz_sessions/onnx-harvest-manifest-cost-golf-v1.csv`
- Markdown manifest: `zzz_sessions/markdown-harvest-manifest-cost-golf-v1.csv`
- Blockers: `zzz_sessions/cost-golf-v1-blockers.md`

## Objective

**Beat the champion for task{NNN} on TOTAL cost: minimize `cost = memory_bytes + params`.** NeuroGolf
score = `25 − ln(memory + params)`. **TARGET +1.0 over the champion (≈ cut total cost to cost/e — a big win;
aim for it, do NOT stop short), but ANY improvement counts** — even one unit of cost is worth submitting.

**Any task is valid and any construction is allowed.** The champion may be any op family, and so may the
candidate — it may replace the champion outright, use a different op family, and carry memory, as long as
`memory + params < champion cost`.

**NEVER stop short / reason from FIRST PRINCIPLES.** "The champion is already optimal / no further trim
found / this route is infeasible" is NOT a valid stop — treat it as under-explored (lazy stop, too-weak
basis, missed trick) and try a genuinely DIFFERENT family. Re-derive the task's rule and the cheapest
representation yourself.

The full method is the **`COST_GOLF_STARTER.txt`** playbook (the round-1 message; repo root), with
`skills-offline/TRICKS.md` as the catalogue to draw on. No op family or construction is privileged — the
champion's shape does not constrain the candidate's.

## Hard Rules

- **MODEL MUST BE `5.6 Sol Pro`.** Before creating ANY session, open the composer model picker (Chrome tool) and
  confirm `5.6 Sol Pro` is selected; select it if not. If `5.6 Sol Pro` is unavailable or rate-limited, fall back to
  `5.6 Sol Extra High` for that session — never drop below `5.6 Sol Extra High`. Re-confirm whenever the picker
  may have reset.
- **Lower total cost `memory_bytes + params`.** The candidate must be a sincere, CORRECT solver (do not
  passthrough/metric-hack). It may be any op family and may carry memory, as long as `memory + params` is
  below the champion's cost.
- **Genuinely SOLVE — run
  `python skills-offline/evaluate/scripts/evaluate.py --input <onnx> --tasks task{NNN} --new-samples 0`** and
  pass 100% of the examples shipped in `skills-offline/evaluate/scripts/neurogolf-2026/task{NNN}.json`
  (train + test + arc-gen). **That is the whole bar — if it passes, ship it**, using any construction that
  reaches the highest score. Never submit an obviously-wrong / identity graph.
- **Do NOT use sparse-stored initializers** — the scorer rejects them (a dead end per the playbook).
- **Every emitted candidate MUST be a downloadable ONNX named `task{NNN}_m{memory}_p{params}.onnx`**, where
  `memory` = intermediate-tensor bytes and `params` = initializer element count, both from **static graph
  inspection only** — no execution. The suffix IS the claimed score `25 − ln(memory + params)`; a candidate
  with no downloadable suffixed ONNX is not harvestable.
- **Harvest gate:** download any ONNX whose `_m{memory}_p{params}` filename **claims to beat the champion** —
  claimed score `= 25 − ln(memory + params)` > champion score from `zzz_sessions/board_total.py <task>` (a ~1.2 s
  `/leaderboard_api` DB read; re-read per pass). Claimed ≤ champion → do NOT download; log why. Markdown is
  ALWAYS harvested every round, regardless of score.
- **Folder routing at harvest** (the session's own confidence, NOT score):
  - `safe/`: passes 100% of the shipped examples (`evaluate.py`).
  - `unsafe/`: fails >=1 shipped example (a genuine attempt that isn't fully correct) — still worth a cheaper
    submit. NEVER route a known-wrong passthrough here — do not submit those at all.
  - `invalid`/`timeout` are assigned only by the submitter/reconciler from the Space verdict.
- Send **10 total messages per task**: Round 1 plus nine follow-up nudges. Track the sent count in the
  session mapping `Rounds` column (N/10). Once a task reaches `10/10`, harvest any remaining artifacts, mark
  it complete, and skip it unless the user explicitly adds more rounds.
- Use a **3700 second submit timeout** for submissions (the modal/Space eval timeout is ~1h — wait the full window).
- Never block the round-robin on one bad session. Log blockers and move on.
- `~/Downloads` is shared. Always `snap` before clicking download cards and `move` immediately after.
- **2-min cooldown = ≥120 s between message SENDS, and the harvest counts toward it.** There is ONE place
  you wait: right before sending a task's message, *after* you've clicked that task's downloads. Check
  how long since the **previous** send and wait only `max(0, 120 − seconds_since_last_send)` (background
  sleep). In round 2+ that is usually **0** (the harvest already spanned >120 s). Round-1 creates have no
  harvest before the send, so consecutive creates wait ~the full 120 s. Never blind-sleep a fixed 120 s
  after every send. Stamp each send's wall-clock time (`Last sent`).

## Setup

Create missing local infrastructure:

```bash
mkdir -p claude-for-chrome-cost-golf-v1/{safe,unsafe,invalid,timeout}
mkdir -p claude-for-chrome-cost-golf-v1-markdown zzz_sessions/cost-golf-v1-baselines
```

Start pollers from the repo root. These drain files produced by browser harvests:

```bash
PY=/Users/geremieyeo/anaconda3/envs/qgentic-ai/bin/python

$PY auto_post_md.py \
  --watch-dir claude-for-chrome-cost-golf-v1-markdown \
  --state-file auto_post_md_state_claude_for_chrome_cost_golf_v1.json \
  --repo golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/cost-golf-v1-post.log 2>&1 &

$PY auto_submit_onnx.py \
  --watch-dir claude-for-chrome-cost-golf-v1/safe \
  --submitted-by cost-golf-v1 \
  --max-age-seconds 0 \
  --submit-timeout 3700 \
  --state-file auto_submit_onnx_state_claude_for_chrome_cost_golf_v1.json \
  --space golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/cost-golf-v1-submit-safe.log 2>&1 &

$PY auto_submit_onnx.py \
  --watch-dir claude-for-chrome-cost-golf-v1/unsafe \
  --submitted-by cost-golf-v1-unsafe \
  --max-age-seconds 0 \
  --submit-timeout 3700 \
  --state-file auto_submit_onnx_state_claude_for_chrome_cost_golf_v1.json \
  --space golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/cost-golf-v1-submit-unsafe.log 2>&1 &
```

**Start all three pollers (1 markdown poster + 2 ONNX submitters) BEFORE any browser action** — they drain asynchronously while you drive. Launch each as a plain `… &`; do NOT combine `nohup … &` with a backgrounded harness call (launch as plain `&` and poll the logs). Confirm each log prints its `Watching ...` startup line before browser work.

## Browser Workflow

Use the claude-in-chrome MCP/browser tool. For each task in
`neurogolf_cost_golf_v1_tasklist.txt`:

1. **Select the right browser/account first:** `list_connected_browsers` → confirm the correct Chrome (the cost-golf-v1 project the user specified) → `select_browser`/`switch_browser` → reuse the project tab. Then open or create the task's session there.
2. Harvest the previous completed round before sending the next message.
3. Download all markdown cards for the task.
4. Download ONNX candidates whose `_m{memory}_p{params}` filename claims to beat the champion (see Harvest gate).
5. Send exactly one next prompt, increment `Rounds`, then move to the next task under the cooldown.

**Browser gotchas (apply every round):**
- **Before sending, open the "Pasted Text" chip** to confirm the right `taskNNN` is inside — the shared macOS clipboard can be clobbered by a parallel driver. After sending, **verify the user bubble appeared**.
- The round-1 prompt is a long paste — make sure the **FULL text landed** in the composer and the user message **actually posted** before moving on; if not, re-open the composer, ensure the full text is present, send again, verify the bubble.
- **Markdown harvest:** ChatGPT lazy-loads older turns **only on a real wheel scroll** — scroll to the very TOP first, then download each card top→bottom (chronological), then run the md mover.
- **READ `GPT56_BUTTON_PATTERNS.md` (repo root) BEFORE harvesting — it is the guide for clicking the ONNX download buttons.** GPT-5.6/5.5 renders the download control in several distinct shapes (prefixed link, bare chip, champion-skip, descriptive) plus INERT non-downloadable chips; the doc shows how to recognize and click each shape and which to skip. **ONNX cards download reliably only via a coordinate click**, not ref/JS clicks.
- **Best ONNX not downloadable** (no card / inert link / `/tmp`-only path) and it claims to beat the champion → do ONE recovery probe: send `Give me the downloadable best ONNX file`, wait ~2 min (the probe is a send, so that wait is the cooldown), then download it or add a `cost-golf-v1-blockers.md` row. Don't loop.
- **ZIP fallback:** ONNX/markdown delivered only inside a zip → download it, `unzip -o -j '*.onnx'` (or `'*.md'`), move each into its folder as `task{NNN}_<name>.onnx`/`.md`, append a manifest row. **Generic-name markdown** ("no md matching taskNNN") → inspect content for the task number and `mv` it in.
- **Checkpoint** the mapping (and the driver memory) every ~10 sends; the manifests + mapping are the source of truth, and both pollers content-hash dedup, so re-harvest/re-submit/re-post is idempotent.

Sweep exactly **10 rounds** over the task list — at most one message per task that has not reached `10/10`.
Do not send an eleventh message unless the user changes the budget. If the project URL/session mapping is
missing, create sessions in the project URL the user provides and record the new `/c/<session_id>` URL.

## Mover Commands

Run from repo root.

```bash
PY=/Users/geremieyeo/anaconda3/bin/python

# ONNX: snap before clicking, move after. Labels: safe | unsafe | skip
$PY .claude/skills/neurogolf-cost-golf-v1/scripts/harvest_move_cost_golf_v1.py snap
$PY .claude/skills/neurogolf-cost-golf-v1/scripts/harvest_move_cost_golf_v1.py move <task> <sid> <label...>

# Markdown: snap before clicking, move after.
$PY .claude/skills/neurogolf-cost-golf-v1/scripts/harvest_move_md_cost_golf_v1.py snap
$PY .claude/skills/neurogolf-cost-golf-v1/scripts/harvest_move_md_cost_golf_v1.py move <task> <sid>
```

If the ONNX mover reports a label/file mismatch, stop and recount downloads. Do not force it.

## Prompt Templates

Round 1 prompt — **the message IS the file `COST_GOLF_STARTER.txt`** (repo root), NOT inlined here:

- **Round 1 (create):** the **full contents of `COST_GOLF_STARTER.txt`** (repo root), with every `{NNN}` replaced by the zero-padded task number (e.g. `task058`). Read the file, substitute, and paste the whole thing into the composer as one message. **THERE IS NO EXTRA TEXT.**

Round 2+ nudge:

```text
Try harder! there is a new baseline uploaded — rebuild the champion skills-offline/solutions_py/task{NNN}.py and read its (possibly lower) cost = memory_bytes + params, then lower task{NNN}'s TOTAL cost MORE (aim for +1.0 = cost → cost/e, but ANY lower-cost win counts). You stopped short: re-derive the task rule from FIRST PRINCIPLES and try a genuinely DISTINCT construction — the champion's op family does not constrain yours, and you may throw its structure away entirely. Draw on skills-offline/TRICKS.md as the catalogue, and iterate cheaply in NumPy before committing a candidate to ONNX. Still CORRECT — 100% of the shipped examples via `python skills-offline/evaluate/scripts/evaluate.py --input <onnx> --tasks task{NNN} --new-samples 0` (that is the whole bar) — no sparse initializers. ALWAYS save a downloadable ONNX named task{NNN}_m{memory}_p{params}.onnx (memory+params from static inspection) with a LOWER memory+params than before, plus a short markdown (champion cost, your m/p, what you built). +1.0 is 100% possible — it is your skill issue. The best ONNX experts in the world have done it. Try an aggressive rewrite if needed.
```

Scorer nudge (only if the Space rejected the last ONNX as unloadable/malformed):

```text
Your last ONNX failed to load or infer under the scorer/onnxruntime path. Fix the graph so it at least LOADS and produces an `output` tensor of the right shape under onnxruntime 1.24.4 (make it load and run under onnxruntime 1.24.4), then resume lowering total cost memory+params toward the champion +1.0 target. Keep the task{NNN}_m{memory}_p{params}.onnx filename.
```

## Reconcile

After pollers have submitted files, reconcile by Space verdict:

```bash
PY=/Users/geremieyeo/anaconda3/bin/python
$PY .claude/skills/neurogolf-cost-golf-v1/scripts/reconcile_submissions_cost_golf_v1.py --dry-run
$PY .claude/skills/neurogolf-cost-golf-v1/scripts/reconcile_submissions_cost_golf_v1.py --apply
```

Expected routing:

- `ok`: leave where it is (it beat the champion and the Space accepted it).
- `unsafe` or `wrong`: move to `unsafe/`.
- `error` or `eval_error`: move to `invalid/`.
- `timeout`: move to `timeout/`.

Note: an `ok` in `unsafe/` stays unsafe (no promotion); a safe-labeled file the Space marks `unsafe`/`error` MOVES (the Space caught what the session missed — dedup may skip re-posting under the unsafe author, which the reconciler flags).

## Verification

```bash
PY=/Users/geremieyeo/anaconda3/envs/qgentic-ai/bin/python
# pollers alive
ps aux | grep -E 'auto_post_md|auto_submit_onnx' | grep -v grep | grep cost-golf-v1
# folder counts
for d in safe unsafe invalid timeout; do printf "%s " $d; ls claude-for-chrome-cost-golf-v1/$d/*.onnx 2>/dev/null | wc -l; done
ls claude-for-chrome-cost-golf-v1-markdown/*.md 2>/dev/null | wc -l
# posted so far (content-unique)
$PY -c "import json;print(len(json.load(open('auto_post_md_state_claude_for_chrome_cost_golf_v1.json')).get('done',[])))"
```
