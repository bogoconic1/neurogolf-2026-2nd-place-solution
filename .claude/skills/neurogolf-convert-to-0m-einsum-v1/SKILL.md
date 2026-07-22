---
name: neurogolf-convert-to-0m-einsum-v1
description: >-
  Actively DRIVE the NeuroGolf convert-to-0m-einsum-v1 sweep. Use when asked to run, resume, create sessions
  for, harvest, submit, or reconcile the einsum-CONVERSION Claude-in-Chrome NeuroGolf sweep. This lane
  takes each task's champion — which is NOT a 0-mem single Einsum — and re-expresses the task as ONE
  input→output `Einsum` emitting `output` directly (memory=0) that BEATS the champion (params < champion
  memory+params; score = 25 − ln(params)). Uses the CONVERT_0M_STARTER.txt param-golf playbook,
  claude-for-chrome-convert-to-0m-einsum-v1 artifact folders, convert-to-0m-einsum-v1 manifests, a 3700s submit
  timeout, and convert-to-0m-einsum-v1 submitted_by labels.
---

# NeuroGolf convert-to-0m-einsum-v1 — "convert a non-0m champion into a single memory-0 Einsum" sweep driver

Each valid task's champion is **NOT** a 0-mem single Einsum (it has memory>0, or is 0-mem but multi-node).
This lane **re-expresses the task as ONE input→output `Einsum` that emits `output` directly** (memory=0) and
**beats the champion** — you win when your `params < champion (memory + params)`. It uses the
`CONVERT_0M_STARTER.txt` param-golf playbook to keep params low; since memory=0, score = `25 − ln(params)`,
and dropping a high-memory champion's memory term is big headroom.

Local artifact roots:

- ONNX: `claude-for-chrome-convert-to-0m-einsum-v1/{safe,unsafe,invalid,timeout}`
- Markdown: `claude-for-chrome-convert-to-0m-einsum-v1-markdown/`
- Baseline builds for analysis only: `zzz_sessions/convert-to-0m-einsum-v1-baselines/`

State/manifests:

- Task list: `zzz_sessions/neurogolf_convert_to_0m_einsum_v1_tasklist.txt` (champions that are NOT 0-mem single-Einsum — convert each to a single 0-mem Einsum that beats it)
- Session map: `zzz_sessions/neurogolf_convert_to_0m_einsum_v1_session_task_mapping.md`
- ONNX manifest: `zzz_sessions/onnx-harvest-manifest-convert-to-0m-einsum-v1.csv`
- Markdown manifest: `zzz_sessions/markdown-harvest-manifest-convert-to-0m-einsum-v1.csv`
- Blockers: `zzz_sessions/convert-to-0m-einsum-v1-blockers.md`

## Objective

**Convert task{NNN}'s champion — which is NOT memory-0 — into ONE input→output `Einsum` that emits the graph
tensor `output` DIRECTLY (→ memory=0), and BEAT the champion.** A single output-direct Einsum has memory=0
(the `output` tensor is excluded from activation memory), so score = `25 − ln(params)`. Build the champion
`skills-offline/solutions_py/task{NNN}.py`, read its memory `m` and params `p`; you **BEAT it when your
`params < m + p`** — you drop the champion's WHOLE memory term, so a high-memory champion has large headroom.
**TARGET +1.0 over the champion (aim for it, do NOT stop short), but ANY beat counts.**

**Valid tasks are ONLY champions that are NOT already a 0-mem single Einsum** (the tasklist is pre-filtered to
these 218). The candidate MUST be a SINGLE input→output Einsum with **memory 0** — do NOT add helper ops,
staging, or non-zero memory. Read the champion to learn the task RULE, but do NOT reuse its non-0m structure —
re-express the rule as one output-direct Einsum.

**NEVER stop short / reason from FIRST PRINCIPLES.** "The champion can't be a single 0-mem Einsum / no
lower-param form / infeasible" is NOT a valid stop — treat it as under-explored and try a genuinely DIFFERENT
family. Re-derive the task's rule and the cheapest single-Einsum representation yourself.

The param-golf method is the **`CONVERT_0M_STARTER.txt`** playbook (the round-1 message; repo root) (10 tricks + an
8-step search order): reuse one initializer name (charged once, referenced many times with different
subscripts); reuse the free graph input; spend equation length to delete stored tensors; factor dense tables
into low-rank basis + adapters; split product axes; collapse role-specific tables into one shared basis;
compress color tables to 10×2/10×3; sweep latent width down; delete tiny helpers; dedupe equal tensors.

## Hard Rules

- **MODEL MUST BE `Pro Extended`.** Before creating ANY session, open the composer model picker (Chrome tool) and
  confirm `Pro Extended` is selected; select it if not. If `Pro Extended` is not offered on the account, fall back to
  `Extra High` for that session — never drop below `Extra High`. Re-confirm whenever the picker may have reset.
- **Produce ONE input→output `Einsum`, memory=0** (emit `output` directly). The candidate must be a sincere,
  CORRECT solver (re-express the task rule, do not passthrough/metric-hack) and stay a single Einsum with
  memory 0; beat the champion's `m + p` on params.
- **Genuinely SOLVE — run `evaluate/scripts/evaluate.py`** and be correct on the known/public examples before
  submitting. You need not perfectly pass *fresh*, but never submit an obviously-wrong / identity graph.
- **Do NOT use sparse-stored initializers** — the scorer rejects them (a dead end per the playbook).
- **Every emitted candidate MUST be a downloadable ONNX named `task{NNN}_m0_p{params}.onnx`** (`m0` because
  memory stays 0). `params` = total initializer element count from **static graph inspection only** — no
  execution. The suffix IS the claimed score `25 − ln(params)`; a candidate with no downloadable suffixed
  ONNX is not harvestable.
- **Harvest gate:** download any ONNX whose `_m0_p{params}` filename **claims to beat the champion** — claimed
  score `= 25 − ln(params)` > champion score from `zzz_sessions/board_total.py <task>` (a ~1.2 s
  `/leaderboard_api` DB read; re-read per pass). Claimed ≤ champion → do NOT download; log why. Markdown is
  ALWAYS harvested every round, regardless of score.
- **Folder routing at harvest** (the session's own confidence, NOT score):
  - `safe/`: `evaluate` ran AND the candidate is correct on the known examples.
  - `unsafe/`: evaluated but imperfect on fresh (a genuine attempt that does not fully generalize).
    NEVER route a known-wrong passthrough here — do not submit those at all.
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
mkdir -p claude-for-chrome-convert-to-0m-einsum-v1/{safe,unsafe,invalid,timeout}
mkdir -p claude-for-chrome-convert-to-0m-einsum-v1-markdown zzz_sessions/convert-to-0m-einsum-v1-baselines
```

Start pollers from the repo root. These drain files produced by browser harvests:

```bash
PY=/Users/geremieyeo/anaconda3/envs/qgentic-ai/bin/python

$PY auto_post_md.py \
  --watch-dir claude-for-chrome-convert-to-0m-einsum-v1-markdown \
  --state-file auto_post_md_state_claude_for_chrome_convert_to_0m_einsum_v1.json \
  --repo golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/convert-to-0m-einsum-v1-post.log 2>&1 &

$PY auto_submit_onnx.py \
  --watch-dir claude-for-chrome-convert-to-0m-einsum-v1/safe \
  --submitted-by convert-to-0m-einsum-v1 \
  --max-age-seconds 0 \
  --submit-timeout 3700 \
  --state-file auto_submit_onnx_state_claude_for_chrome_convert_to_0m_einsum_v1.json \
  --space golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/convert-to-0m-einsum-v1-submit-safe.log 2>&1 &

$PY auto_submit_onnx.py \
  --watch-dir claude-for-chrome-convert-to-0m-einsum-v1/unsafe \
  --submitted-by convert-to-0m-einsum-v1-unsafe \
  --max-age-seconds 0 \
  --submit-timeout 3700 \
  --state-file auto_submit_onnx_state_claude_for_chrome_convert_to_0m_einsum_v1.json \
  --space golfingteam10000pts/neurogolf-leaderboard-v2 \
  > zzz_sessions/convert-to-0m-einsum-v1-submit-unsafe.log 2>&1 &
```

**Start all three pollers (1 markdown poster + 2 ONNX submitters) BEFORE any browser action** — they drain asynchronously while you drive. Launch each as a plain `… &`; do NOT combine `nohup … &` with a backgrounded harness call (launch as plain `&` and poll the logs). Confirm each log prints its `Watching ...` startup line before browser work.

## Browser Workflow

Use the claude-in-chrome MCP/browser tool. For each task in
`neurogolf_convert_to_0m_einsum_v1_tasklist.txt`:

1. **Select the right browser/account first:** `list_connected_browsers` → confirm the correct Chrome (the convert-to-0m-einsum-v1 project the user specified) → `select_browser`/`switch_browser` → reuse the project tab. Then open or create the task's session there.
2. Harvest the previous completed round before sending the next message.
3. Download all markdown cards for the task.
4. Download ONNX candidates whose `_m0_p{params}` filename claims to beat the champion (see Harvest gate).
5. Send exactly one next prompt, increment `Rounds`, then move to the next task under the cooldown.

**Browser gotchas (apply every round):**
- **Before sending, open the "Pasted Text" chip** to confirm the right `taskNNN` is inside — the shared macOS clipboard can be clobbered by a parallel driver. After sending, **verify the user bubble appeared**.
- The round-1 prompt is a long paste — make sure the **FULL text landed** in the composer and the user message **actually posted** before moving on; if not, re-open the composer, ensure the full text is present, send again, verify the bubble.
- **Markdown harvest:** ChatGPT lazy-loads older turns **only on a real wheel scroll** — scroll to the very TOP first, then download each card top→bottom (chronological), then run the md mover.
- **ONNX cards download reliably only via a coordinate click**, not ref/JS clicks.
- **Best ONNX not downloadable** (no card / inert link / `/tmp`-only path) and it claims to beat the champion → do ONE recovery probe: send `Give me the downloadable best ONNX file`, wait ~2 min (the probe is a send, so that wait is the cooldown), then download it or add a `convert-to-0m-einsum-v1-blockers.md` row. Don't loop.
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
$PY .claude/skills/neurogolf-convert-to-0m-einsum-v1/scripts/harvest_move_convert_to_0m_einsum_v1.py snap
$PY .claude/skills/neurogolf-convert-to-0m-einsum-v1/scripts/harvest_move_convert_to_0m_einsum_v1.py move <task> <sid> <label...>

# Markdown: snap before clicking, move after.
$PY .claude/skills/neurogolf-convert-to-0m-einsum-v1/scripts/harvest_move_md_convert_to_0m_einsum_v1.py snap
$PY .claude/skills/neurogolf-convert-to-0m-einsum-v1/scripts/harvest_move_md_convert_to_0m_einsum_v1.py move <task> <sid>
```

If the ONNX mover reports a label/file mismatch, stop and recount downloads. Do not force it.

## Prompt Templates

Round 1 prompt — **the message IS the file `CONVERT_0M_STARTER.txt`** (repo root), NOT inlined here (same as neurogolf-v7 pastes `starter_message.txt`):

- **Round 1 (create):** the **full contents of `CONVERT_0M_STARTER.txt`** (repo root), with every `{NNN}` replaced by the zero-padded task number (e.g. `task058`). Read the file, substitute, and paste the whole thing into the composer as one message. **THERE IS NO EXTRA TEXT.**

Round 2+ nudge:

```text
Try harder! there is a new baseline uploaded — rebuild the champion skills-offline/solutions_py/task{NNN}.py and read its memory m and params p, then convert task{NNN} into a SINGLE input→output Einsum emitting `output` directly (memory=0) with params BELOW m+p (aim for +1.0, but ANY beat counts). You stopped short: re-derive the task rule from FIRST PRINCIPLES and try a genuinely DISTINCT family (if skills-offline/kaggle_hidden_zero_candidates/task{NNN}_*.onnx exists it is a VERIFIED-ZERO past attempt — study it and do NOT reproduce its structure; the graded test is hidden fresh samples, so a lookup keyed to the public examples scores 0). Concrete moves (re-apply the param-golf playbook you were given in round 1): sweep the latent width LOWER (k=8,6,5,4,3,2 — the first correct width is rarely minimal); factor the largest dense table into a low-rank basis + adapters; collapse role-specific tables into ONE shared basis; compress any color table to 10x2/10x3; expand the equation / reuse ONE initializer name harder to DELETE a stored tensor; delete or fold every tiny 2- or 4-element helper. Keep it ONE input→output Einsum with memory=0, still CORRECT (run skills-offline/evaluate/scripts/evaluate.py), no sparse initializers. ALWAYS save a downloadable ONNX named task{NNN}_m0_p{params}.onnx (params from static inspection) with params below m+p, plus a short markdown (champion m+p, your p, tricks used, equation). Try aggressive rewrites to prevent getting stuck in local optima.
```

Scorer nudge (only if the Space rejected the last ONNX as unloadable/malformed):

```text
Your last ONNX failed to load or infer under the scorer/onnxruntime path. Fix the graph so it at least LOADS and produces an `output` tensor of the right shape under onnxruntime 1.24.4 (make it load and run under onnxruntime 1.24.4), keeping it ONE input→output Einsum with memory=0, then resume converting task{NNN} into a lower-param 0-mem Einsum that beats the champion. Keep the task{NNN}_m0_p{params}.onnx filename.
```

## Reconcile

After pollers have submitted files, reconcile by Space verdict:

```bash
PY=/Users/geremieyeo/anaconda3/bin/python
$PY .claude/skills/neurogolf-convert-to-0m-einsum-v1/scripts/reconcile_submissions_convert_to_0m_einsum_v1.py --dry-run
$PY .claude/skills/neurogolf-convert-to-0m-einsum-v1/scripts/reconcile_submissions_convert_to_0m_einsum_v1.py --apply
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
ps aux | grep -E 'auto_post_md|auto_submit_onnx' | grep -v grep | grep convert-to-0m-einsum-v1
# folder counts
for d in safe unsafe invalid timeout; do printf "%s " $d; ls claude-for-chrome-convert-to-0m-einsum-v1/$d/*.onnx 2>/dev/null | wc -l; done
ls claude-for-chrome-convert-to-0m-einsum-v1-markdown/*.md 2>/dev/null | wc -l
# posted so far (content-unique)
$PY -c "import json;print(len(json.load(open('auto_post_md_state_claude_for_chrome_convert_to_0m_einsum_v1.json')).get('done',[])))"
```
