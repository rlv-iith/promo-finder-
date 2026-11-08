"""End-to-end: episodes -> candidates -> (optional LLM judge) -> ranked promos."""
from __future__ import annotations

import re

from .candidates import Promo, ShowContext, nms, single_windows
from .parse import Episode
from .stitch import stitched_cross_episode, stitched_same_episode

STOP = {"the", "and", "was", "who", "with", "her", "his", "for", "she", "him", "scripts", "episode"}


STITCH_MARGIN = 0.03   # a multi-segment promo must beat the single clips it replaces by this much


def generate(episodes: list[Episode], ctx: ShowContext, per_episode: int = 3, stitch: bool = True) -> list[Promo]:
    singles: list[Promo] = []
    stitches: list[Promo] = []
    for i, ep in enumerate(episodes):
        singles += nms(single_windows(ep, ctx), per_episode=per_episode)
        if stitch:
            cands = stitched_same_episode(ep, ctx)
            if i + 1 < len(episodes) and episodes[i + 1].number == ep.number + 1:
                cands += stitched_cross_episode(ep, episodes[i + 1], ctx)
            stitches += nms(cands, per_episode=2)
    kept = nms(singles, max_overlap=0.5)
    for s in sorted(stitches, key=lambda p: p.score, reverse=True):
        clash = [k for k in kept if s.overlap(k) > 0.35 * min(s.duration, k.duration)]
        if any(len(k.segments) > 1 for k in clash):
            continue
        if all(s.score >= k.score + STITCH_MARGIN for k in clash):
            kept = [k for k in kept if k not in clash] + [s]
    # cap per starting episode again after stitches moved things around
    return nms(kept, max_overlap=0.5, per_episode=per_episode)


def why_line(p: Promo) -> str:
    """One plain-English line built from the signals that fired."""
    def first(reasons, keys):
        for r in reasons:
            k, _, v = r.partition(":")
            if k in keys:
                return keys[k].format(v=v)
        return None
    hook = first(p.reasons.get("hook", []), {
        "violence": "opens on on-screen violence", "reveal": "opens on a reveal",
        "threat": "opens on a threat", "insult": "opens on a public humiliation",
        "stakes": "opens with the stakes stated", "delivery": "opens on a heated line"}) or "opens mid-action"
    cliff = first(p.reasons.get("cliff", []), {
        "open question": "ends on an unanswered question", "line cut mid-thought": "cuts off mid-sentence",
        "cue": "ends on a held, unresolved beat", "threat": "ends on a threat that hasn't landed yet",
        "reveal": "ends right at a reveal", "violence": "ends mid-danger"}) or "ends at a turn"
    roles = p.reasons.get("roles") or []
    readable = f"roles are spelled out ({', '.join(roles[:3])}) so it reads cold" if roles else "needs little backstory"
    if p.reasons.get("unexplained_names") and len(p.reasons["unexplained_names"]) >= 3:
        readable = "but leans on names a new viewer won't know"
    return f"It {hook}, {cliff}; {readable}."


def title_keywords(show_title: str) -> set[str]:
    words = re.findall(r"[a-z]+", show_title.lower())
    return {w for w in words if len(w) > 3 and w not in STOP}


def rank_for_posting(promos: list[Promo], show_title: str, top_n: int = 15) -> list[Promo]:
    """Order to post in. Quality first, then two editorial nudges:
    - funnel: clips from early episodes pay off fast for a viewer who taps through to E1
    - title promise: clips that show what the title promises ("soldier", "duke") sell the premise
    and diversity: one promo per episode in the top list."""
    kw = title_keywords(show_title)
    for p in promos:
        ep = p.segments[0].episode
        funnel = 0.04 if ep <= 5 else 0.02 if ep <= 15 else 0.0
        text = " ".join(e.text.lower() for e in p.events)
        promise = 0.02 if any(re.search(rf"\b{k}", text) for k in kw) else 0.0
        penalty = 0.03 * sum(f.startswith(("opens mid", "many unexplained", "dead air")) for f in p.flags)
        p.scores["post_priority"] = round(p.score + funnel + promise - penalty, 4)
        p.scores["funnel_bonus"], p.scores["title_promise_bonus"] = funnel, promise
    ordered = sorted(promos, key=lambda p: p.scores["post_priority"], reverse=True)
    picked, seen = [], set()
    for p in ordered:
        ep = p.segments[0].episode
        if ep in seen:
            continue
        picked.append(p)
        seen.add(ep)
        if len(picked) == top_n:
            break
    return picked
