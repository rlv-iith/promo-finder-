"""Cheap, explainable per-event signals (no AI). Everything here is plain rules
driven by lexicon.json, so an editor can read *why* a moment scored high."""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

from .parse import Event

LEXICON_PATH = Path(__file__).with_name("lexicon.json")


@lru_cache(maxsize=4)
def load_lexicon(path: str = str(LEXICON_PATH)) -> dict[str, re.Pattern]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pats = {}
    for key, stems in raw.items():
        if key.startswith("_"):
            continue
        if key == "continuation_starts":
            pats[key] = re.compile(r"^(?:" + "|".join(stems) + ")", re.I)
        else:
            pats[key] = re.compile(r"\b(?:" + "|".join(stems) + ")", re.I)
    return pats


def hits(pat: re.Pattern, text: str) -> list[str]:
    return [m.group(0).strip().lower() for m in pat.finditer(text or "")]


def event_heat(e: Event, lex: dict) -> tuple[float, list[str]]:
    """How 'loud' a single moment is, 0..~1.5, plus the reasons."""
    if e.kind == "camera":
        return 0.0, []
    text, why = e.text, []
    score = {"dialogue": 0.15, "action": 0.10, "sfx": 0.05}.get(e.kind, 0.0)
    # Violence you can SEE (action/sfx) beats violence someone talks about.
    violence_w = 0.65 if e.kind in ("action", "sfx") else 0.30
    weights = [("violence", violence_w), ("reveal", 0.45), ("betrayal", 0.45), ("threat", 0.40),
               ("insult", 0.30), ("stakes", 0.15)]
    for key, w in weights:
        h = hits(lex[key], text)
        if h:
            score += w
            why.append(f"{key}:{h[0]}")
    tags = " ".join(e.tags)
    hot = len(hits(lex["hot_tags"], tags))
    cool = len(hits(lex["cool_tags"], tags))
    score += min(hot, 2) * 0.15 - min(cool, 2) * 0.12
    if hot:
        why.append(f"delivery:{','.join(e.tags[:2])}")
    if e.kind == "dialogue":
        if "!" in text:
            score += 0.12
        if text.rstrip().endswith("?"):
            score += 0.08
        if len(text.split()) <= 8 and ("!" in text or "?" in text):
            score += 0.05  # short, punchy lines play best in a 3-second window
    if "V.O." in e.tags:
        score -= 0.15      # inner monologue rarely stops a scroll
    return max(0.0, min(score, 1.5)), why


def is_continuation(text: str, lex: dict) -> bool:
    return bool(lex["continuation_starts"].match(text.strip().lstrip("\"'“ ")))


def cliff_marks(e: Event, lex: dict) -> tuple[float, list[str]]:
    """Does this moment leave something unresolved? (+) vs. resolve/soften it (-)."""
    text = e.text + " " + " ".join(e.tags)
    s, why = 0.0, []
    if e.kind == "dialogue":
        t = e.text.rstrip()
        if t.endswith("?"):
            s += 0.35; why.append("open question")
        if t.endswith("...") or t.endswith("—") or t.endswith("-"):
            s += 0.30; why.append("line cut mid-thought")
        if t.endswith("!"):
            s += 0.10
    m = hits(lex["cliff_marks"], text)
    if m:
        s += 0.35; why.append(f"cue:{m[0]}")
    for key in ("threat", "reveal", "violence"):
        h = hits(lex[key], e.text)
        if h:
            s += 0.20; why.append(f"{key}:{h[0]}")
            break
    r = hits(lex["resolution_marks"], text)
    if r:
        s -= 0.45; why.append(f"resolves:{r[0]}")
    if lex["cool_tags"].search(" ".join(e.tags)):
        s -= 0.15
    return s, why
