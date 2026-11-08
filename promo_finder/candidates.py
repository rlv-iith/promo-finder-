"""Generate and score promo candidates with rules only.

A candidate is 1+ segments (episode, start, end). Cuts are only allowed on shot
boundaries and never inside a line of dialogue, so every clip is editable as-is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .parse import Episode, Event, fmt
from .signals import cliff_marks, event_heat, hits, is_continuation, load_lexicon

HOOK_WINDOW = 3.0       # brief: "something in the first 3 seconds"
CLIFF_WINDOW = 5.0      # how much of the ending we judge
MIN_LEN, MAX_LEN = 20.0, 60.0


@dataclass
class Segment:
    episode: int
    start: float
    end: float

    @property
    def dur(self) -> float:
        return self.end - self.start

    def to_dict(self):
        return {"episode": self.episode, "start": fmt(self.start), "end": fmt(self.end),
                "start_s": round(self.start, 2), "end_s": round(self.end, 2)}


@dataclass
class Promo:
    segments: list[Segment]
    events: list[Event]
    scores: dict = field(default_factory=dict)
    hook: Event | None = None
    cliff: Event | None = None
    reasons: dict = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)
    kind: str = "single"      # single | stitched-same-episode | stitched-cross-episode
    llm: dict | None = None

    @property
    def duration(self) -> float:
        return sum(s.dur for s in self.segments)

    @property
    def score(self) -> float:
        return self.scores.get("final", self.scores.get("rules", 0.0))

    @property
    def key(self) -> str:
        return "+".join(f"E{s.episode}@{fmt(s.start)}-{fmt(s.end)}" for s in self.segments)

    def overlap(self, other: "Promo") -> float:
        """Seconds of shared footage."""
        tot = 0.0
        for a in self.segments:
            for b in other.segments:
                if a.episode == b.episode:
                    tot += max(0.0, min(a.end, b.end) - max(a.start, b.start))
        return tot


class ShowContext:
    """Show-level facts the scorer needs: lexicon, character names."""

    def __init__(self, episodes: list[Episode], lexicon_path: str | None = None):
        self.lex = load_lexicon(lexicon_path) if lexicon_path else load_lexicon()
        self.episodes = {e.number: e for e in episodes}
        # Only one-word speaker labels are names ("JULIANA"); "BLOND NOBLEMAN" is a description.
        speakers = {ev.speaker for ep in episodes for ev in ep.events
                    if ev.kind == "dialogue" and len(ev.speaker.split()) == 1 and len(ev.speaker) > 2}
        line_counts: dict[str, int] = {}
        for ep in episodes:
            for ev in ep.events:
                if ev.kind == "dialogue":
                    line_counts[ev.speaker] = line_counts.get(ev.speaker, 0) + 1
        # Leads = the two characters with the most lines. Promos should sell *their* story.
        self.leads = {s for s, _ in sorted(line_counts.items(), key=lambda x: -x[1])[:2]}
        # Capitalised words used mid-sentence 3+ times are treated as names/places.
        counts: dict[str, int] = {}
        for ep in episodes:
            for ev in ep.events:
                if ev.kind != "dialogue":
                    continue
                for m in re.finditer(r"(?<=[a-z,;] )([A-Z][a-z]{2,})", ev.text):
                    counts[m.group(1)] = counts.get(m.group(1), 0) + 1
        common = {"Father", "Mother", "Lady", "Lord", "Your", "Grace", "Duke", "Duchess", "God", "Crown",
                  "King", "Honor", "Sir", "The", "Please", "Look", "Her", "His", "My"}
        roles = {"MAID", "GUARD", "JUDGE", "PRIEST", "BUTLER", "FOOTMAN", "NOBLEMAN", "NOBLEWOMAN",
                 "INFORMANT", "CONSTABLE", "CONSPIRATOR", "HOUSEKEEPER", "TOWNSMAN", "LOYALIST", "LORD", "STRANGER"}
        self.names = {w.upper() for w, c in counts.items() if c >= 3 and w not in common} | \
                     {s for s in speakers if s.isalpha() and s not in roles}
        self.heat = {}
        for ep in episodes:
            for ev in ep.events:
                self.heat[(ep.number, ev.idx)] = event_heat(ev, self.lex)


def _cut_points(ep: Episode) -> tuple[list[float], list[float]]:
    dialogue = [e for e in ep.events if e.kind == "dialogue"]

    def inside_line(t: float) -> bool:
        return any(d.start + 0.05 < t < d.end - 0.05 for d in dialogue)

    shots = [e for e in ep.events if e.kind == "camera"] or ep.events
    starts = sorted({round(e.start, 2) for e in shots} | {0.0})
    ends = sorted({round(e.end, 2) for e in shots} | {round(ep.duration, 2)})
    return [t for t in starts if not inside_line(t)], [t for t in ends if not inside_line(t)]


def events_in(ep: Episode, start: float, end: float) -> list[Event]:
    return [e for e in ep.events if e.start >= start - 0.01 and e.start < end - 0.01]


def score_promo(p: Promo, ctx: ShowContext, at_episode_end: bool = False) -> Promo:
    lex = ctx.lex
    evs = [e for e in p.events if e.kind != "camera"]
    if not evs:
        p.scores = dict.fromkeys(["rules", "hook", "cliff", "standalone", "intensity", "density", "lead"], 0.0)
        return p
    first_seg, last_seg = p.segments[0], p.segments[-1]
    H = lambda e: ctx.heat[(e.episode, e.idx)]

    # ---- hook: loudest moment that starts in the first 3 seconds
    opening = [e for e in evs if e.episode == first_seg.episode and e.start < first_seg.start + HOOK_WINDOW]
    hook_ev = max(opening, key=lambda e: H(e)[0], default=None)
    hook = H(hook_ev)[0] if hook_ev else 0.0
    hook_why = H(hook_ev)[1] if hook_ev else ["nothing happens in first 3s"]
    first_line = next((e for e in evs if e.kind == "dialogue"), None)
    if first_line and first_line.start < first_seg.start + 6 and is_continuation(first_line.text, lex):
        hook -= 0.30
        p.flags.append("opens mid-conversation")
    hook_n = max(0.0, min(hook / 1.1, 1.0))

    # ---- cliffhanger: does the ending leave something open?
    tail = [e for e in evs if e.episode == last_seg.episode and e.end > last_seg.end - CLIFF_WINDOW]
    last_line = next((e for e in reversed(evs) if e.kind == "dialogue"), None)
    cm = [cliff_marks(e, lex) for e in tail] or [(0.0, [])]
    best_tail = max(cm, key=lambda x: x[0])
    ll = cliff_marks(last_line, lex) if last_line else (0.0, [])
    cliff = 0.5 * best_tail[0] + 0.5 * ll[0] + (0.35 * H(last_line)[0] if last_line else 0)
    cliff_ev = last_line
    if not tail:
        cliff -= 0.25
        p.flags.append("dead air at the end")
    if at_episode_end and cliff > 0:
        cliff += 0.15           # writers put the cliffhanger at the episode's end
    if any("resolves" in w for w in best_tail[1] + ll[1]):
        p.flags.append("ending may feel resolved")
    cliff_n = max(0.0, min((cliff + 0.3) / 1.3, 1.0))

    # ---- standalone: would a stranger follow it?
    text = " ".join(e.text for e in evs if e.kind == "dialogue")
    speakers = {tok for e in evs if e.kind == "dialogue" for tok in e.speaker.split()}
    referenced = {w.upper() for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text) if w.upper() in ctx.names}
    unexplained = referenced - speakers
    roles = set(hits(lex["self_explaining"], text))
    backrefs = hits(lex["backstory_refs"], text)
    standalone = 0.6 + min(len(roles), 5) * 0.05 - max(0, len(unexplained) - 1) * 0.08 - min(len(backrefs), 3) * 0.08
    if "opens mid-conversation" in p.flags:
        standalone -= 0.25
    # Starting on a natural beat (scene start / after a pause) reads better cold than mid-exchange.
    ep0 = ctx.episodes.get(first_seg.episode)
    if ep0 is not None:
        prev = [e for e in ep0.events if e.kind == "dialogue" and e.end <= first_seg.start + 0.05]
        first_ev = evs[0]
        if not prev or first_seg.start - prev[-1].end >= 2.0 or prev[-1].scene != first_ev.scene:
            standalone += 0.08
        elif first_seg.start - prev[-1].end < 0.8:
            standalone -= 0.05
            p.flags.append("starts mid-exchange")
    if len(unexplained) >= 4:
        p.flags.append(f"many unexplained names: {', '.join(sorted(unexplained))}")
    standalone_n = max(0.0, min(standalone, 1.0))

    # ---- intensity & pacing
    heats = sorted((H(e)[0] for e in evs if e.kind in ("dialogue", "action", "sfx")), reverse=True)
    top = heats[: max(3, len(heats) // 2)]
    heat_n = min((sum(top) / len(top)) / 0.85, 1.0) if top else 0.0
    covered = 0.0
    for seg in p.segments:
        spans = sorted((max(e.start, seg.start), min(e.end, seg.end)) for e in evs
                       if e.episode == seg.episode and e.kind in ("dialogue", "action"))
        cur_s = cur_e = None
        for s, e in spans:
            if cur_e is None or s > cur_e:
                if cur_e is not None:
                    covered += cur_e - cur_s
                cur_s, cur_e = s, e
            else:
                cur_e = max(cur_e, e)
        if cur_e is not None:
            covered += cur_e - cur_s
    density = min(covered / max(p.duration, 1e-6), 1.0)
    if density < 0.45:
        p.flags.append("slow / lots of silence")

    d = p.duration
    dur_prior = 1.0 if 25 <= d <= 50 else (0.85 + 0.15 * (d - 20) / 5 if d < 25 else 1.0 - 0.15 * (d - 50) / 10)

    # ---- does the heroine/hero carry it? (villain-only scheming scenes sell less)
    lead_lines = sum(1 for e in evs if e.kind == "dialogue" and e.speaker in ctx.leads)
    lead_named = any(l.title() in e.text for e in evs if e.kind == "action" for l in ctx.leads)
    lead = 1.0 if lead_lines >= 2 else (0.6 if lead_lines or lead_named else 0.0)
    if lead == 0.0:
        p.flags.append("leads absent (villain/side-plot scene)")

    rules = (0.28 * hook_n + 0.28 * cliff_n + 0.14 * standalone_n + 0.14 * heat_n + 0.08 * density
             + 0.08 * lead) * dur_prior
    # The brief says promos are *usually* one continuous stretch: each jump cut must earn its place.
    n_jumps = len(p.segments) - 1
    cross = len({s.episode for s in p.segments}) > 1
    rules -= 0.04 * n_jumps + (0.03 if cross else 0.0)
    p.scores = {"rules": round(rules, 4), "hook": round(hook_n, 3), "cliff": round(cliff_n, 3),
                "standalone": round(standalone_n, 3), "intensity": round(heat_n, 3), "density": round(density, 3),
                "lead": lead}
    p.hook, p.cliff = hook_ev, cliff_ev
    p.reasons = {"hook": hook_why, "cliff": best_tail[1] + ll[1], "roles": sorted(roles),
                 "unexplained_names": sorted(unexplained)}
    return p


def single_windows(ep: Episode, ctx: ShowContext, min_len=MIN_LEN, max_len=MAX_LEN) -> list[Promo]:
    starts, ends = _cut_points(ep)
    out = []
    for s in starts:
        for e in ends:
            if e - s < min_len:
                continue
            if e - s > max_len:
                break
            evs = events_in(ep, s, e)
            p = Promo([Segment(ep.number, s, e)], evs)
            out.append(score_promo(p, ctx, at_episode_end=abs(e - ep.duration) < 0.5))
    return out


def nms(promos: list[Promo], max_overlap: float = 0.35, per_episode: int | None = None) -> list[Promo]:
    """Greedy non-max suppression: keep the best, drop near-duplicates."""
    kept: list[Promo] = []
    count: dict[int, int] = {}
    for p in sorted(promos, key=lambda x: x.score, reverse=True):
        if any(p.overlap(k) > max_overlap * min(p.duration, k.duration) for k in kept):
            continue
        ep = p.segments[0].episode
        if per_episode and count.get(ep, 0) >= per_episode:
            continue
        kept.append(p)
        count[ep] = count.get(ep, 0) + 1
    return kept
