"""Is the tool better than what a naive editor or script would grab?

Baselines (same 3 picks per episode, same 20-60 s length, same clean-cut rules):
  random   - 3 random non-overlapping legal clips per episode (averaged over many seeds)
  ending   - the last ~35 s of each episode (writers put the cliffhanger there)
  opening  - the first ~35 s of each episode (often a recap / cold open)
All are scored with the exact metric used for the tool (eval/evaluate.py).

  python eval/baselines.py --input <show folder or zip> --tool outputs --save outputs/validation/baselines.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from evaluate import evaluate  # noqa: E402
from promo_finder.candidates import MAX_LEN, MIN_LEN, _cut_points  # noqa: E402
from promo_finder.parse import load_show  # noqa: E402

TARGET = 35.0


def as_record(ep, s, e, rank, kind):
    return {"id": f"{kind}-E{ep:02d}-{rank}", "kind": kind, "rank_in_episode": rank,
            "segments": [{"episode": ep, "start_s": s, "end_s": e}]}


def legal_windows(ep):
    starts, ends = _cut_points(ep)
    return [(s, e) for s in starts for e in ends if MIN_LEN <= e - s <= MAX_LEN]


def closest(windows, length, anchor, at_end):
    key = (lambda w: abs(w[1] - anchor) + abs((w[1] - w[0]) - length) * 0.1) if at_end else \
          (lambda w: abs(w[0] - anchor) + abs((w[1] - w[0]) - length) * 0.1)
    return min(windows, key=key)


def ending(eps, per_ep=1):
    out = []
    for ep in eps:
        w = legal_windows(ep)
        if w:
            s, e = closest(w, TARGET, ep.duration, True)
            out.append(as_record(ep.number, s, e, 1, "ending"))
    return out


def opening(eps):
    out = []
    for ep in eps:
        w = legal_windows(ep)
        if w:
            s, e = closest(w, TARGET, 0.0, False)
            out.append(as_record(ep.number, s, e, 1, "opening"))
    return out


def random_picks(eps, rng, per_ep=3):
    out = []
    for ep in eps:
        w = legal_windows(ep)
        rng.shuffle(w)
        kept = []
        for s, e in w:
            if all(min(e, b) - max(s, a) <= 0 for a, b in kept):
                kept.append((s, e))
            if len(kept) == per_ep:
                break
        out += [as_record(ep.number, s, e, r, "random") for r, (s, e) in enumerate(kept, 1)]
    return out


def summary(promos, gold, split):
    r = evaluate(promos, gold, split)
    return {"strict": r["strict_found"], "top1": r["is_top1"], "iou": r["mean_iou"],
            "lenient": r["found_in_top3"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tool", default="outputs")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--save")
    a = ap.parse_args()
    src = Path(a.input)
    if src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp()); zipfile.ZipFile(src).extractall(tmp); src = tmp
    eps = load_show(src)
    gold = Path(__file__).with_name("gold.json")
    tool = json.loads((Path(a.tool) / "promos.json").read_text(encoding="utf-8"))["promos"]

    rows = {}
    for split in ("test", None):
        name = split or "all"
        rows[name] = {"tool (3 per episode)": summary(tool, gold, split),
                      "episode ending (1 per episode)": summary(ending(eps), gold, split),
                      "episode opening (1 per episode)": summary(opening(eps), gold, split)}
        runs = [summary(random_picks(eps, random.Random(seed)), gold, split) for seed in range(a.seeds)]
        agg = {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}
        st = sorted(r["strict"] for r in runs)
        agg["strict_p95"] = st[int(0.95 * (len(st) - 1))]
        tool_strict = rows[name]["tool (3 per episode)"]["strict"]
        agg["share_of_random_runs_matching_tool"] = sum(r["strict"] >= tool_strict for r in runs) / len(runs)
        rows[name]["random (3 per episode, %d seeds)" % a.seeds] = agg
    for name, table in rows.items():
        print(f"\n{name.upper()} split")
        for k, v in table.items():
            extra = (f"  [95th pct {v['strict_p95']:.0%}; random >= tool in {v['share_of_random_runs_matching_tool']:.1%} of runs]"
                     if "strict_p95" in v else "")
            print(f"  {k:34s} strict: {v['strict']:5.0%}  #1 strict: {v['top1']:5.0%}  mean IoU: {v['iou']:.2f}  "
                  f"lenient: {v['lenient']:4.0%}{extra}")
    if a.save:
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        Path(a.save).write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
