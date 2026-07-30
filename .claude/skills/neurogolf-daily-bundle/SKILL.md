---
name: neurogolf-daily-bundle
description: >-
  Rebuild the shared NeuroGolf champion-baseline bundle. Download the latest safe champions from the v2
  HF Space, regenerate the solutions_py builders, and rebuild skills-offline.zip — ONE bundle shared by
  every driven project. Use when asked to run/refresh the daily bundle or rebuild skills-offline.zip.
  Runs locally (headless: API + shell). Companion to the runtime-v1 + large-einsum-v1/v2/v3 driver skills.
---

# NeuroGolf — daily bundle (skills-offline.zip rebuild)

Rebuild the one champion-baseline bundle (`skills-offline.zip`) that gives the solver sessions their
context. This is a single headless script — no browser step.

**`previous-attempts` is NOT part of this** — notes/past attempts are no longer used as LLM context.
Markdown is still harvested + posted to HF separately (that's unchanged).

## Run the bundle script

```bash
~/anaconda3/envs/qgentic-ai/bin/python \
  .claude/skills/neurogolf-daily-bundle/scripts/daily_v2_bundle.py
```

What it does (idempotent, safe to re-run):
1. **Download** the latest safe champions (`submission.zip`, 400 ONNXes) + the champion leaderboard
   metadata (`/leaderboard_table`) from the v2 Space to temp paths, and **regenerate**
   `skills-offline/solutions_py/taskNNN.py` from them via `export_dense_builders.py` (the dense champion
   builders ride inside `skills-offline.zip`). No champion zip is kept in the bundle.
2. **Rebuild** `skills-offline.zip` at repo root.

Flags: `--no-download` (zip only), `--no-zip` (download only). `HF_TOKEN` is read from `.env` and
never printed.

The rebuilt artifact:
- `skills-offline.zip` at repo root (the script resolves the repo from its own location).

Uploading the zip into each project's ChatGPT **Sources** is a manual step the user does by hand (not
part of this skill).

## Scheduling

Headless, so a **cloud routine** (`/schedule`) or **Desktop Scheduled Task** can run it standalone.
Re-running is a cheap no-op (idempotent).

## Verification

- `ls -la skills-offline.zip` — freshly rebuilt (recent mtime).
- `unzip -l skills-offline.zip | grep -c solutions_py/task` — the day's champion builders (400) are inside.
- KAGGLE-line sum spot-check: `python -c "import glob,re;print(round(sum(float(re.search(r\"'score':\s*([0-9.]+)\",open(f).read()).group(1)) for f in glob.glob('skills-offline/solutions_py/task*.py')),4))"`
  should match the live board `overall_score`.
