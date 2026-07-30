#!/usr/bin/env python
"""Read the v2 leaderboard (a cheap DB read, ~1.2s, NOT an evaluation — no Space/OOM load).

  python zzz_sessions/board_total.py                 # overall total (stop-target check)
  python zzz_sessions/board_total.py task255 256     # per-task CHAMPION scores (the download gate)
  python zzz_sessions/board_total.py --all           # all 400 task champion scores (cache per pass)

Per-task mode prints "taskNNN <score>" lines — compare a session's CLAIMED score to this to decide
whether to download its onnx (download only if claimed > champion; see the neurogolf-v3/v4 skills).
The per-task score is the EFFECTIVE champion = max(safe champion, slow-runtime manual champion): a
`status='slow'` sidecar candidate that beats the safe board score is treated as PROMOTED here, so the
download gate never harvests something already worse than the known-best slow champion (this matches the
daily bundle, which folds those slow champions into solutions_py). Pass --safe-only for the raw safe
board score. HF_TOKEN comes from the environment. A long httpx timeout + retry survives the pollers.
"""
import os, json, sys, time
from gradio_client import Client

def fetch():
    last = None
    for _ in range(4):
        try:
            c = Client("golfingteam10000pts/neurogolf-leaderboard-v2",
                       token=os.environ.get("HF_TOKEN"), httpx_kwargs={"timeout": 120}, verbose=False)
            return json.loads(c.predict(api_name="/leaderboard_api"))
        except Exception as e:
            last = e; time.sleep(3)
    raise SystemExit(f"leaderboard read failed: {type(last).__name__}: {last}")

d = fetch()
flags = [a for a in sys.argv[1:] if a.startswith("-")]
args = [a for a in sys.argv[1:] if not a.startswith("-")]

if not args and "--all" not in flags:
    ws = d.get("overall_score_with_slow")
    ws_txt = f" overall_score_with_slow={ws:.4f}" if ws is not None else ""
    print(f"overall_score={d['overall_score']:.4f}{ws_txt} resolved={d['resolved_count']} unsafe={d['unsafe_count']}")
else:
    def eff(t):
        safe = t.get("score") or 0
        if "--safe-only" in flags:
            return safe
        return max(safe, t.get("slow_score") or 0)  # slow champion beating safe counts as PROMOTED
    score = {t["task_id"]: eff(t) for t in d["tasks"]}
    if "--all" in flags:
        keys = sorted(score)
    else:
        keys = [a if a.startswith("task") else f"task{int(a):03d}" for a in args]
    for k in keys:
        print(f"{k} {score.get(k, 0):.4f}")
