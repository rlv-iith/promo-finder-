"""Score tool output against hand-labelled reference promos.

A reference promo counts as FOUND if some tool promo shares >= 50% of the
shorter clip's footage with it. We report recall within the tool's picks for
that episode, and whether it is the episode's #1 pick.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder.parse import to_seconds  # noqa: E402


def load_gold(path):
    g = json.loads(Path(path).read_text(encoding="utf-8"))["promos"]
    for p in g:
        p["segs"] = [(int(e), to_seconds(s), to_seconds(t)) for e, s, t in p["segments"]]
        p["dur"] = sum(t - s for _, s, t in p["segs"])
    return g


def overlap(a_segs, b_segs):
    return sum(max(0.0, min(t1, t2) - max(s1, s2))
               for e1, s1, t1 in a_segs for e2, s2, t2 in b_segs if e1 == e2)


def evaluate(promos: list[dict], gold_path, split=None) -> dict:
    """promos: dicts from promos.json (segments with episode/start_s/end_s, rank_in_episode).

    Strict match = IoU >= 0.5 (shared footage / combined footage), so a long or
    random clip that merely contains the moment does not count.
    """
    gold = [g for g in load_gold(gold_path) if split in (None, g["split"])]
    rows = []
    for g in gold:
        best, best_iou, top1_iou, loose = None, 0.0, 0.0, False
        for p in promos:
            segs = [(s["episode"], s["start_s"], s["end_s"]) for s in p["segments"]]
            dur = sum(t - s for _, s, t in segs)
            ov = overlap(segs, g["segs"])
            if ov <= 0:
                continue
            iou = ov / (dur + g["dur"] - ov)
            if ov / min(dur, g["dur"]) >= 0.5:
                loose = True
            if p["rank_in_episode"] == 1 and p["segments"][0]["episode"] == g["segs"][0][0]:
                top1_iou = max(top1_iou, iou)
            if iou > best_iou:
                best, best_iou = p, iou
        err = None
        if best is not None:
            bs, be = best["segments"][0]["start_s"], best["segments"][-1]["end_s"]
            err = round(abs(bs - g["segs"][0][1]) + abs(be - g["segs"][-1][2]), 1)
        rows.append({"id": g["id"], "split": g["split"], "episode": g["segs"][0][0],
                     "loose_found": loose,
                     "strict_found": best_iou >= 0.5,
                     "top1_strict": top1_iou >= 0.5,
                     "best_iou": round(best_iou, 2),
                     "boundary_error_s": err,
                     "tool_clip": best["id"] if best else None})
    n = len(rows) or 1
    errs = [r["boundary_error_s"] for r in rows if r["strict_found"]]
    return {"n": len(rows),
            "found_in_top3": sum(r["loose_found"] for r in rows) / n,     # old, lenient metric
            "strict_found": sum(r["strict_found"] for r in rows) / n,
            "is_top1": sum(r["top1_strict"] for r in rows) / n,
            "mean_iou": sum(r["best_iou"] for r in rows) / n,
            "median_boundary_error_s": sorted(errs)[len(errs) // 2] if errs else None,
            "rows": rows}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir", nargs="?", default="outputs")
    ap.add_argument("--final", action="store_true", help="also score the held-out test split")
    ap.add_argument("--save", help="write results JSON here")
    a = ap.parse_args()
    promos = json.loads((Path(a.out_dir) / "promos.json").read_text(encoding="utf-8"))["promos"]
    gold_path = Path(__file__).with_name("gold.json")
    splits = ("dev", "test", None) if a.final else ("dev",)
    results = {}
    for split in splits:
        r = evaluate(promos, gold_path, split)
        results[split or "all"] = r
        print(f"{split or 'all':5s} n={r['n']:2d}  strict match (IoU>=0.5): {r['strict_found']:.0%}   "
              f"#1 pick strict: {r['is_top1']:.0%}   mean IoU: {r['mean_iou']:.2f}   "
              f"median boundary error: {r['median_boundary_error_s']}s   (lenient overlap: {r['found_in_top3']:.0%})")
    for row in results["all" if a.final else "dev"]["rows"]:
        print(row)
    if a.save:
        Path(a.save).write_text(json.dumps(results, indent=2), encoding="utf-8")
