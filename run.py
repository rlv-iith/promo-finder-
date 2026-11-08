#!/usr/bin/env python3
"""Find promos in a microdrama show.

  python run.py --input "path/to/show_folder_or.zip" --out outputs
  python run.py --input show.zip --provider gemini --model gemini-2.5-flash      # + LLM judge
  python run.py --input show.zip --provider openai --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import tempfile
import time
import zipfile
from pathlib import Path

from promo_finder.candidates import ShowContext
from promo_finder.llm import PROVIDERS, judge
from promo_finder.parse import load_show
from promo_finder.pipeline import generate, rank_for_posting
from promo_finder.report import write_all


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="folder of episode .txt files, or a .zip of them")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--title", help="show title (default: folder/zip name)")
    ap.add_argument("--per-episode", type=int, default=3, help="max promos kept per episode")
    ap.add_argument("--min-score", type=float, default=0.60, help="drop promos below this rules score (set on dev episodes)")
    ap.add_argument("--top", type=int, default=15, help="length of the post-first list")
    ap.add_argument("--no-stitch", action="store_true", help="single continuous clips only")
    ap.add_argument("--lexicon", help="custom genre lexicon JSON")
    ap.add_argument("--provider", default="none", choices=["none", *PROVIDERS])
    ap.add_argument("--model", default="gemini-2.5-flash")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--llm-top-k", type=int, default=60, help="how many top candidates the LLM judges")
    a = ap.parse_args(argv)

    t0 = time.time()
    src = Path(a.input)
    if src.suffix.lower() == ".zip":
        tmp = Path(tempfile.mkdtemp())
        zipfile.ZipFile(src).extractall(tmp)
        folder = tmp
    else:
        folder = src
    title = a.title or src.stem.replace("-scripts", "").split(" (")[0]
    episodes = load_show(folder)
    if not episodes:
        sys.exit(f"No episode scripts found in {src}")
    print(f"Loaded {len(episodes)} episodes ({sum(e.duration for e in episodes) / 60:.1f} min) of '{title}'")

    ctx = ShowContext(episodes, a.lexicon)
    promos = generate(episodes, ctx, per_episode=a.per_episode, stitch=not a.no_stitch)
    promos = [p for p in promos if p.scores["rules"] >= a.min_score]
    print(f"Rules stage: {len(promos)} promos kept (score >= {a.min_score})")

    mode = "rules only"
    usage = {}
    if a.provider != "none":
        prov = PROVIDERS[a.provider](model=a.model, base_url=a.base_url, cache_dir=str(Path(a.out) / ".llm_cache"))
        premise = f"'{title}'. Episode 1: {episodes[0].logline}"
        print(f"LLM stage: judging top {a.llm_top_k} with {a.provider}:{a.model}")
        judge(promos, prov, premise, ctx=ctx, top_k=a.llm_top_k)
        mode = f"rules + LLM judge ({a.provider}:{a.model})"
        usage = {"api_calls": prov.calls, "cache_hits": prov.cache_hits,
                 "approx_tokens_in": prov.tokens_in, "approx_tokens_out": prov.tokens_out}

    posting = rank_for_posting(promos, title, top_n=a.top)
    meta = {"show": title, "run_at": dt.datetime.now().isoformat(timespec="seconds"), "mode": mode,
            "episodes": [e.number for e in episodes], "runtime_s": round(time.time() - t0, 1),
            "settings": {k: v for k, v in vars(a).items() if k != "input"}, "llm_usage": usage}
    write_all(Path(a.out), promos, posting, meta, [e.number for e in episodes])
    print(f"Done in {meta['runtime_s']}s -> {a.out}/report.md, promos.json, promos.csv")


if __name__ == "__main__":
    main()
