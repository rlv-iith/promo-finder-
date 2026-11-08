"""Write raw outputs: promos.json (everything), promos.csv (one row per promo), report.md (readable)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from .candidates import Promo
from .parse import fmt
from .pipeline import why_line


def promo_dict(p: Promo, pid: str, rank_in_ep: int) -> dict:
    llm = p.llm or {}
    hook_q = llm.get("hook_line") or (p.hook.text if p.hook else "")
    cliff_q = llm.get("cliffhanger_line") or (p.cliff.text if p.cliff else "")
    return {
        "id": pid,
        "kind": p.kind,
        "rank_in_episode": rank_in_ep,
        "segments": [s.to_dict() for s in p.segments],
        "duration_s": round(p.duration, 2),
        "hook": {"at": f"E{p.hook.episode} {fmt(p.hook.start)}" if p.hook else None,
                 "who": (p.hook.speaker or p.hook.kind.upper()) if p.hook else None, "line": hook_q},
        "cliffhanger": {"at": f"E{p.cliff.episode} {fmt(p.cliff.start)}" if p.cliff else None,
                        "who": (p.cliff.speaker or p.cliff.kind.upper()) if p.cliff else None, "line": cliff_q},
        "why": llm.get("why") or why_line(p),
        "scores": p.scores,
        "signals": p.reasons,
        "flags": p.flags,
        "llm": {k: v for k, v in llm.items() if k not in ("hook_line", "cliffhanger_line", "why")} or None,
        "transcript": [f"E{e.episode} {fmt(e.start)} {e.speaker or e.kind.upper()}: {e.text}"
                       for e in p.events if e.kind != "camera"],
    }


def write_all(out_dir: Path, promos: list[Promo], posting: list[Promo], meta: dict, n_episodes: list[int]) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    by_ep: dict[int, list[Promo]] = {}
    for p in promos:
        by_ep.setdefault(p.segments[0].episode, []).append(p)
    records, ids = [], {}
    for ep in sorted(by_ep):
        for r, p in enumerate(sorted(by_ep[ep], key=lambda x: x.score, reverse=True), 1):
            pid = f"E{ep:02d}-P{r}"
            ids[id(p)] = pid
            records.append(promo_dict(p, pid, r))
    post_ids = [ids[id(p)] for p in posting]
    rec_by_id = {r["id"]: r for r in records}
    for i, pid in enumerate(post_ids, 1):
        rec_by_id[pid]["post_order"] = i

    (out_dir / "promos.json").write_text(json.dumps({"meta": meta, "post_first": post_ids, "promos": records},
                                                    indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out_dir / "promos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "post_order", "kind", "segments", "duration_s", "score", "hook_at", "hook_line",
                    "cliff_at", "cliff_line", "why", "flags"])
        for r in records:
            segs = " + ".join(f"E{s['episode']} {s['start']}-{s['end']}" for s in r["segments"])
            w.writerow([r["id"], r.get("post_order", ""), r["kind"], segs, r["duration_s"],
                        r["scores"].get("final", r["scores"]["rules"]), r["hook"]["at"], r["hook"]["line"],
                        r["cliffhanger"]["at"], r["cliffhanger"]["line"], r["why"], "; ".join(r["flags"])])

    L = [f"# Promo picks: {meta['show']}", "",
         f"Run: {meta['run_at']} · mode: {meta['mode']} · episodes: {len(n_episodes)} · "
         f"promos found: {len(records)} · runtime: {meta['runtime_s']}s", "",
         "## Post these first", "",
         "| # | Promo | Segments | Len | Hook (first 3s) | Cliffhanger | Why |", "|---|---|---|---|---|---|---|"]
    for i, pid in enumerate(post_ids, 1):
        r = rec_by_id[pid]
        segs = "<br>".join(f"E{s['episode']} {s['start']}–{s['end']}" for s in r["segments"])
        L.append(f"| {i} | {pid} | {segs} | {r['duration_s']:.0f}s | {r['hook']['line'][:90]} | "
                 f"{r['cliffhanger']['line'][:90]} | {r['why']} |")
    L += ["", "## All promos by episode", ""]
    empty = [e for e in n_episodes if e not in by_ep]
    for r in records:
        segs = " + ".join(f"E{s['episode']} {s['start']}–{s['end']}" for s in r["segments"])
        flag = f"  ⚠ {'; '.join(r['flags'])}" if r["flags"] else ""
        L.append(f"- **{r['id']}** ({r['kind']}, {r['duration_s']:.0f}s, score {r['scores'].get('final', r['scores']['rules']):.3f}) "
                 f"{segs}  \n  Hook: “{r['hook']['line']}” · Cliff: “{r['cliffhanger']['line']}”  \n  {r['why']}{flag}")
    if empty:
        L += ["", f"Episodes with no promo above threshold: {', '.join(map(str, empty))}"]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    return records
