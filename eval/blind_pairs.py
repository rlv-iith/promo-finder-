"""Blind A/B test: tool's #1 promo per episode vs a baseline clip from the SAME episode.

  make:   python eval/blind_pairs.py make  --input <show> --tool outputs --out outputs/validation
  score:  python eval/blind_pairs.py score --out outputs/validation --votes votes_a.json [votes_b.json ...]

`make` writes:
  pairs_blind.json   - what a judge sees: two anonymised transcripts per pair (A/B order randomised,
                       times shown from 0 s, no episode numbers, no scores)
  pairs_key.json     - which side is the tool (kept away from judges)
  blind_test.html    - the same pairs as a click-through page for human judges (works offline)
Votes format: {"P01": "A", "P02": "B", ...}  ("tie" allowed)
"""
from __future__ import annotations

import argparse
import html
import json
import math
import random
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
from baselines import closest, legal_windows, TARGET  # noqa: E402
from promo_finder.parse import load_show  # noqa: E402


def clip_lines(ep, s, e):
    out = []
    for ev in ep.events:
        if ev.kind == "camera" or ev.start < s - 0.01 or ev.start >= e - 0.01:
            continue
        who = ev.speaker.title() if ev.kind == "dialogue" else ev.kind.upper()
        out.append(f"{ev.start - s:5.1f}s  {who}: {ev.text}")
    return out


def iou(a, b):
    ov = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    return ov / ((a[1] - a[0]) + (b[1] - b[0]) - ov)


def make(a):
    src = Path(a.input)
    if src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp()); zipfile.ZipFile(src).extractall(tmp); src = tmp
    eps = {e.number: e for e in load_show(src)}
    tool = json.loads((Path(a.tool) / "promos.json").read_text(encoding="utf-8"))["promos"]
    rng = random.Random(a.seed)
    pairs, key, skipped = [], {}, []
    for rec in tool:
        if rec["rank_in_episode"] != 1 or len(rec["segments"]) != 1:
            continue                      # single-clip #1 picks only, for a like-for-like comparison
        seg = rec["segments"][0]
        ep = eps[seg["episode"]]
        t = (seg["start_s"], seg["end_s"])
        windows = legal_windows(ep)
        baselines = {"ending": closest(windows, TARGET, ep.duration, True)}
        far = [w for w in windows if iou(w, t) == 0 and abs((w[1] - w[0]) - (t[1] - t[0])) < 8] or \
              [w for w in windows if iou(w, t) == 0]
        if far:
            baselines["random"] = rng.choice(far)
        for bname, b in baselines.items():
            if iou(b, t) >= 0.5:
                skipped.append({"episode": ep.number, "baseline": bname, "reason": "tool picked the same clip"})
                continue
            tool_side = rng.choice("AB")
            sides = {tool_side: t, ("B" if tool_side == "A" else "A"): b}
            pid = f"P{len(pairs) + 1:02d}"
            pairs.append({"id": pid,
                          "A": {"seconds": round(sides["A"][1] - sides["A"][0]), "lines": clip_lines(ep, *sides["A"])},
                          "B": {"seconds": round(sides["B"][1] - sides["B"][0]), "lines": clip_lines(ep, *sides["B"])}})
            key[pid] = {"tool": tool_side, "baseline": bname, "episode": ep.number, "tool_clip": rec["id"],
                        "baseline_clip": [round(b[0], 2), round(b[1], 2)]}
    # shuffle presentation order so baseline type isn't guessable from position
    order = list(range(len(pairs))); rng.shuffle(order)
    pairs = [pairs[i] for i in order]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "pairs_blind.json").write_text(json.dumps({"instructions": RUBRIC, "pairs": pairs}, indent=1,
                                                     ensure_ascii=False), encoding="utf-8")
    (out / "pairs_key.json").write_text(json.dumps({"key": key, "skipped_same_clip": skipped}, indent=1),
                                        encoding="utf-8")
    (out / "blind_test.html").write_text(render_html(pairs), encoding="utf-8")
    print(f"{len(pairs)} blind pairs written ({len(skipped)} skipped because the tool chose the baseline clip itself)")


RUBRIC = ("You are choosing which clip to post as a 20-60 second ad for a vertical microdrama on TikTok/Reels. "
          "Viewers have never seen the show. Pick the clip that would make more people stop scrolling in the first "
          "3 seconds, understand what is going on, and need to watch the next episode. Judge only the transcript.")


def render_html(pairs):
    data = json.dumps(pairs, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Blind Promo Test</title>
<style>
:root{{--bg:#f7f7f5;--card:#fff;--ink:#1d232b;--muted:#5d6670;--line:#dde1e6;--accent:#0b5cad;--pick:#e7f0fb}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15181c;--card:#1d2126;--ink:#e8ebee;--muted:#9aa3ad;--line:#2e343b;--accent:#6aa8ec;--pick:#1f3044}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.45 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}}
main{{max-width:1100px;margin:0 auto;padding:20px 16px 60px}}h1{{font-size:20px;margin:0 0 4px}}
.muted{{color:var(--muted)}}.bar{{display:flex;gap:12px;align-items:center;flex-wrap:wrap;margin:14px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}@media(max-width:760px){{.grid{{grid-template-columns:1fr}}}}
.clip{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;cursor:pointer}}
.clip.sel{{outline:3px solid var(--accent);background:var(--pick)}}.clip h2{{font-size:16px;margin:0 0 8px}}
.clip pre{{white-space:pre-wrap;font:13px/1.45 ui-monospace,Consolas,monospace;margin:0;max-height:52vh;overflow:auto}}
button{{font:inherit;padding:8px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink);cursor:pointer}}
button.primary{{background:var(--accent);color:#fff;border-color:var(--accent)}}input{{font:inherit;padding:7px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--ink)}}
</style></head><body><main>
<h1>Blind Promo Test</h1>
<p class="muted">{html.escape(RUBRIC)} Click the better clip (or Tie). Your votes stay on this computer; press <b>Download votes</b> at the end.</p>
<div class="bar"><label>Your name/initials <input id="who" placeholder="e.g. RK"></label>
<span id="prog" class="muted"></span></div>
<div class="grid"><div class="clip" id="A"></div><div class="clip" id="B"></div></div>
<div class="bar"><button id="prev">← Back</button><button id="tie">Tie</button>
<button id="next" class="primary">Next →</button><button id="dl">Download votes</button></div>
<script>
const P={data};let i=0;const votes={{}};
const $=id=>document.getElementById(id);
function show(){{const p=P[i];for(const s of ["A","B"]){{const el=$(s);
el.innerHTML="<h2>Clip "+s+" · "+p[s].seconds+"s</h2><pre></pre>";el.querySelector("pre").textContent=p[s].lines.join("\\n");
el.classList.toggle("sel",votes[p.id]===s);}}
$("prog").textContent="Pair "+(i+1)+" of "+P.length+" · answered "+Object.keys(votes).length+(votes[P[i].id]==="tie"?" · (tie)":"");}}
function vote(v){{votes[P[i].id]=v;show();setTimeout(()=>{{if(i<P.length-1){{i++;show();}}}},250);}}
$("A").onclick=()=>vote("A");$("B").onclick=()=>vote("B");$("tie").onclick=()=>vote("tie");
$("next").onclick=()=>{{if(i<P.length-1){{i++;show();}}}};$("prev").onclick=()=>{{if(i>0){{i--;show();}}}};
$("dl").onclick=()=>{{const who=($("who").value||"anon").replace(/[^\\w-]/g,"");
const b=new Blob([JSON.stringify(votes,null,1)],{{type:"application/json"}});const a=document.createElement("a");
a.href=URL.createObjectURL(b);a.download="votes_"+who+".json";a.click();}};
show();
</script></main></body></html>"""


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - m) / d, (c + m) / d)


def binom_p_one_sided(k, n):
    """P(X >= k) for X ~ Binomial(n, 0.5): chance of a result this good if the tool were no better."""
    return sum(math.comb(n, j) for j in range(k, n + 1)) / 2 ** n


def score(a):
    out = Path(a.out)
    key = json.loads((out / "pairs_key.json").read_text(encoding="utf-8"))["key"]
    results = {}
    for vf in a.votes:
        votes = json.loads(Path(vf).read_text(encoding="utf-8"))
        for pid, v in votes.items():
            if pid not in key or v not in ("A", "B", "tie"):
                continue
            k = key[pid]
            r = results.setdefault(k["baseline"], {"win": 0, "loss": 0, "tie": 0})
            r["tie" if v == "tie" else "win" if v == k["tool"] else "loss"] += 1
    report = {}
    for b, r in results.items():
        n = r["win"] + r["loss"]
        lo, hi = wilson(r["win"], n)
        report[b] = {**r, "tool_win_rate": r["win"] / n if n else None, "ci95": [round(lo, 2), round(hi, 2)],
                     "p_value_one_sided": round(binom_p_one_sided(r["win"], n), 4)}
        print(f"tool vs {b:7s}: {r['win']} wins, {r['loss']} losses, {r['tie']} ties  ->  tool preferred "
              f"{report[b]['tool_win_rate']:.0%} of decided pairs (95% CI {lo:.0%}-{hi:.0%}, "
              f"p={report[b]['p_value_one_sided']})")
    if a.save:
        Path(a.save).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make"); m.add_argument("--input", required=True); m.add_argument("--tool", default="outputs")
    m.add_argument("--out", default="outputs/validation"); m.add_argument("--seed", type=int, default=7)
    s = sub.add_parser("score"); s.add_argument("--out", default="outputs/validation")
    s.add_argument("--votes", nargs="+", required=True); s.add_argument("--save")
    a = ap.parse_args()
    make(a) if a.cmd == "make" else score(a)
