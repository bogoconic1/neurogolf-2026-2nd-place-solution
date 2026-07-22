#!/usr/bin/env python3
"""Daily NeuroGolf bundle — Half A (download champions -> regenerate builders -> zip skills-offline).

  1. Download the latest safe champions (submission.zip, 400 ONNXes) AND the safe-champion leaderboard
     metadata (from /download_submission + /leaderboard_table) from the v2 Space (to temp paths).
  2. Regenerate skills-offline/solutions_py/taskNNN.py from those champions via
     export_dense_builders.py — the dense Python builders the solver sees as the baseline. The
     leaderboard JSON supplies each builder's champion metadata (score, memory_bytes, params, etc).
  3. Rebuild skills-offline.zip.

This produces ONE bundle (`skills-offline.zip`) shared by every NeuroGolf project. Half B (delete the
project's existing Sources, then upload skills-offline.zip) is computer-use, driven separately via
claude-in-chrome, once PER PROJECT you drive (v3 / v4 / v5) — see the neurogolf-daily-bundle SKILL.md.

NOTE: the champion baseline is no longer shipped as a zip inside the bundle; it ships as the
`solutions_py/` dense builders (run a builder to materialize its taskNNN.onnx). `previous-attempts`
is not part of this workflow — notes/past attempts are not used as LLM context anymore. Markdown is
still harvested and posted to HF separately (auto_post_md); it just isn't merged here.

Usage:
  python daily_v2_bundle.py                 # download + regenerate + zip
  python daily_v2_bundle.py --no-download   # zip only (reuse existing solutions_py)
  python daily_v2_bundle.py --no-zip        # download + regenerate only

HF_TOKEN is read from <repo>/.env (never printed).
"""
import os, sys, json, shutil, subprocess, tempfile

REPO = os.path.expanduser('~/Documents/NeuroGolf-2026')
SPACE = "golfingteam10000pts/neurogolf-leaderboard-v2"
SKILLS_OFFLINE = os.path.join(REPO, 'skills-offline')
SOLUTIONS_PY = os.path.join(SKILLS_OFFLINE, 'solutions_py')
EXPORTER = os.path.join(REPO, 'export_dense_builders.py')


def hf_token():
    for ln in open(os.path.join(REPO, '.env')):
        if ln.startswith('HF_TOKEN='):
            return ln.split('=', 1)[1].strip().strip('"').strip("'")
    raise SystemExit("HF_TOKEN not found in .env")


def build_leaderboard_json(client):
    """Fetch /leaderboard_table and emit the safe-champion metadata JSON the exporter consumes
    (task_id, status=champion, score, memory_bytes, params, candidate_id, submitted_by, updated_at).
    The leaderboard table is the current-champion set: one row per task."""
    tbl = client.predict(api_name="/leaderboard_table")
    H = {h: i for i, h in enumerate(tbl["headers"])}
    def num(s):
        # leaderboard_table cells are now real numbers (datatype=number); pass those through,
        # but still tolerate the old comma-formatted strings.
        if isinstance(s, (int, float)):
            return s
        return (s or "").replace(",", "").strip()
    tasks = []
    for row in tbl["data"]:
        sc = num(row[H["Score"]])
        tasks.append({
            "task_id": row[H["Task"]],
            "status": "champion",
            "score": float(sc) if sc else None,
            "memory_bytes": int(num(row[H["Memory"]]) or 0),
            "params": int(num(row[H["Params"]]) or 0),
            "candidate_id": row[H["Attempt"]] or None,
            "submitted_by": row[H["Submitted by"]] or None,
            "updated_at": row[H["Updated"]] or None,
        })
    path = os.path.join(tempfile.gettempdir(), "neurogolf_leaderboard.json")
    with open(path, "w") as f:
        json.dump({"tasks": tasks}, f)
    return path


def fetch_inputs():
    from gradio_client import Client
    print(f"[1/4] downloading champions + leaderboard from {SPACE} ...")
    client = Client(SPACE, token=hf_token(), verbose=False)
    zip_path = client.predict(api_name="/download_submission")
    print(f"      -> champions {zip_path} ({os.path.getsize(zip_path)//1024} KB)")
    lb_path = build_leaderboard_json(client)
    print(f"      -> leaderboard {lb_path}")
    return zip_path, lb_path


def regenerate_builders(zip_path, leaderboard_json):
    print("[2/4] regenerating solutions_py/ dense builders from champions ...")
    if os.path.isdir(SOLUTIONS_PY):
        shutil.rmtree(SOLUTIONS_PY)
    subprocess.run([
        sys.executable, EXPORTER,
        '--zip', zip_path,
        '--extract-dir', '/tmp/neurogolf_champ_onnx',
        '--out-dir', '/tmp/neurogolf_champ_builders',
        '--flat-py-dir', SOLUTIONS_PY,
        '--leaderboard-json', leaderboard_json,
    ], check=True)
    n = len([f for f in os.listdir(SOLUTIONS_PY)
             if f.startswith('task') and f.endswith('.py')])
    print(f"      -> {SOLUTIONS_PY} ({n} builders)")


def make_zip():
    print("[3/3] zipping skills-offline ...")
    out = shutil.make_archive(os.path.join(REPO, 'skills-offline'), 'zip', root_dir=SKILLS_OFFLINE)
    print(f"      -> {os.path.basename(out)} ({os.path.getsize(out)//1024//1024} MB)")


def main():
    if '--no-download' not in sys.argv:
        zip_path, leaderboard_json = fetch_inputs()
        regenerate_builders(zip_path, leaderboard_json)
    else:
        print("[1-2/3] download + regenerate skipped (--no-download)")
    if '--no-zip' not in sys.argv:
        make_zip()
    else:
        print("[3/3] zipping skipped (--no-zip)")
    print("\nHalf A done. Now run Half B (delete + upload skills-offline.zip to each project's Sources "
          "you drive — v3 / v4 / v5) via claude-in-chrome — see the neurogolf-daily-bundle skill.")


if __name__ == '__main__':
    main()
