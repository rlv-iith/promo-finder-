"""Multi-segment promos: join a strong opening to a strong ending.

Two patterns editors use:
  1. same episode  - cut out the slow middle between a hook and a cliffhanger
  2. cross episode - episode N's climax + the start of N+1, ending on N+1's next open beat
Rules pick the pairs; the shared scorer judges the joined clip like any other.
"""
from __future__ import annotations

import re

from .candidates import MAX_LEN, Promo, Segment, ShowContext, _cut_points, events_in, score_promo
from .parse import Episode

SEG_MIN, SEG_MAX = 8.0, 35.0
TOTAL_MIN = 25.0


def _segments(ep: Episode, ctx: ShowContext, lo: float, hi: float) -> list[Promo]:
    starts, ends = _cut_points(ep)
    out = []
    for s in starts:
        if s < lo or s > hi:
            continue
        for e in ends:
            if e - s < SEG_MIN:
                continue
            if e - s > SEG_MAX or e > hi + SEG_MAX:
                break
            p = Promo([Segment(ep.number, s, e)], events_in(ep, s, e))
            out.append(score_promo(p, ctx, at_episode_end=abs(e - ep.duration) < 0.5))
    return out


def _words(p: Promo) -> set[str]:
    return set(re.findall(r"[a-z']+", " ".join(e.text.lower() for e in p.events if e.kind == "dialogue")))


def _join(a: Promo, b: Promo, ctx: ShowContext, kind: str, at_end: bool) -> Promo | None:
    total = a.duration + b.duration
    if not (TOTAL_MIN <= total <= MAX_LEN):
        return None
    wa, wb = _words(a), _words(b)
    if wa and wb and len(wa & wb) / min(len(wa), len(wb)) > 0.5:
        return None          # episode N+1 often re-plays N's last lines; don't show them twice
    p = Promo(a.segments + b.segments, a.events + b.events, kind=kind)
    return score_promo(p, ctx, at_episode_end=at_end)


def opening_rank(p: Promo) -> float:
    return p.scores["hook"] + 0.5 * p.scores["intensity"] + 0.3 * p.scores["standalone"]


def closing_rank(p: Promo) -> float:
    return p.scores["cliff"] + 0.5 * p.scores["intensity"]


def stitched_same_episode(ep: Episode, ctx: ShowContext, top: int = 6) -> list[Promo]:
    segs = _segments(ep, ctx, 0, ep.duration)
    openers = sorted(segs, key=opening_rank, reverse=True)[:top * 3]
    closers = sorted(segs, key=closing_rank, reverse=True)[:top * 3]
    out = []
    for a in openers[:top]:
        for b in closers:
            gap = b.segments[0].start - a.segments[-1].end
            if gap < 4.0:            # no real gap -> it's just a single clip, handled elsewhere
                continue
            j = _join(a, b, ctx, "stitched-same-episode", abs(b.segments[-1].end - ep.duration) < 0.5)
            if j:
                out.append(j)
    return out


def _continues(prev: Episode, nxt: Episode) -> bool:
    if re.search(r"direct(ly)? (follows|continu)|picks up|moments? after|immediately", nxt.continuity, re.I):
        return True
    last_h = prev.scene_headings.get(max(prev.scene_headings, default=0), "")
    first_h = nxt.scene_headings.get(min(nxt.scene_headings, default=0), "")
    return bool(last_h) and last_h.split("–")[0].strip() == first_h.split("–")[0].strip()


def stitched_cross_episode(prev: Episode, nxt: Episode, ctx: ShowContext, top: int = 5) -> list[Promo]:
    if not _continues(prev, nxt):
        return []
    tails = _segments(prev, ctx, max(0.0, prev.duration - 45), prev.duration)
    heads = _segments(nxt, ctx, 0, 40)
    tails = sorted(tails, key=opening_rank, reverse=True)[:top]
    heads = sorted(heads, key=closing_rank, reverse=True)[:top * 2]
    out = []
    for a in tails:
        for b in heads:
            j = _join(a, b, ctx, "stitched-cross-episode", abs(b.segments[-1].end - nxt.duration) < 0.5)
            if j:
                out.append(j)
    return out
