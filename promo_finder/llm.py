"""Optional LLM judge. The model only *judges and describes* candidates the rules
already found; it never invents timestamps. Any boundary it suggests must point
at a real line ID, and code re-checks length and cut rules before accepting it.

Providers (pick with --provider):
  none     rules only (default, free, offline)
  gemini   Google AI Studio free tier       env GEMINI_API_KEY   --model gemini-2.5-flash
  openai   any OpenAI-compatible endpoint   env OPENAI_API_KEY   --base-url ... --model ...
           (works for OpenAI, Groq, OpenRouter, Together, a local Ollama at http://localhost:11434/v1)
Responses are cached on disk by prompt hash, so re-runs cost nothing and are repeatable.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.request
from pathlib import Path

from .candidates import MAX_LEN, MIN_LEN, Promo
from .parse import fmt

RUBRIC = """You are a senior promo editor for vertical microdrama ads (TikTok/Reels/Shorts).
Judge ONE candidate clip cut from the show's screenplay. Viewers have never seen the show.

Score each 1-5 (5 = best):
- hook: would the first 3 seconds stop a scrolling thumb? (slap, reveal, shocking line = high; scenery, calm talk = low)
- cliffhanger: does it cut at an unresolved moment so the viewer must know what happens next? (resolved/hug/fade-out = low)
- standalone: can a stranger follow who is who and what is at stake from this clip alone?

Rules:
- Quote lines EXACTLY from the transcript; do not invent dialogue, names or events.
- If a better start or end exists INSIDE this transcript, give the line IDs (e.g. "L3"), else null.
- Reply with JSON only, no prose:
{"hook": int, "cliffhanger": int, "standalone": int,
 "hook_line": "exact quote", "cliffhanger_line": "exact quote",
 "why": "one sentence on why this promo works or fails",
 "better_start": "L#" | null, "better_end": "L#" | null,
 "reject": true|false, "reject_reason": "..." }"""


def transcript(p: Promo) -> tuple[str, dict]:
    lines, ids = [], {}
    for i, e in enumerate(x for x in p.events if x.kind != "camera"):
        lid = f"L{i + 1}"
        ids[lid] = e
        who = e.speaker if e.kind == "dialogue" else e.kind.upper()
        tag = f" ({', '.join(e.tags)})" if e.tags else ""
        lines.append(f"{lid} [E{e.episode} {fmt(e.start)}] {who}{tag}: {e.text}")
    return "\n".join(lines), ids


def build_prompt(p: Promo, premise: str) -> tuple[str, dict]:
    t, ids = transcript(p)
    seg = "; ".join(f"E{s.episode} {fmt(s.start)}-{fmt(s.end)}" for s in p.segments)
    return f"{RUBRIC}\n\nSHOW PREMISE (for you only; viewers don't know it): {premise}\n\nCANDIDATE ({seg}, {p.duration:.0f}s):\n{t}", ids


class Provider:
    name = "none"

    def __init__(self, model: str = "", base_url: str = "", cache_dir: str = ".llm_cache", min_interval: float = 4.0):
        self.model, self.base_url = model, base_url
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.min_interval, self._last = min_interval, 0.0
        self.calls = self.cache_hits = self.tokens_in = self.tokens_out = 0

    def complete(self, prompt: str) -> str:
        key = hashlib.sha256(f"{self.name}|{self.model}|{prompt}".encode()).hexdigest()[:24]
        f = self.cache / f"{key}.json"
        if f.exists():
            self.cache_hits += 1
            return json.loads(f.read_text(encoding="utf-8"))["text"]
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)       # stay under free-tier rate limits
        for attempt in range(4):
            try:
                text = self._call(prompt)
                break
            except Exception as ex:  # rate limit / transient network
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
                last = ex  # noqa: F841
        self._last = time.time()
        self.calls += 1
        self.tokens_in += len(prompt) // 4
        self.tokens_out += len(text) // 4
        f.write_text(json.dumps({"model": self.model, "text": text}), encoding="utf-8")
        return text

    def _call(self, prompt: str) -> str:
        raise NotImplementedError

    @staticmethod
    def _post(url: str, body: dict, headers: dict) -> dict:
        req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **headers})
        with urllib.request.urlopen(req, timeout=90) as r:
            return json.loads(r.read().decode())


class Gemini(Provider):
    name = "gemini"

    def _call(self, prompt: str) -> str:
        key = os.environ["GEMINI_API_KEY"]
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={key}"
        out = self._post(url, {"contents": [{"parts": [{"text": prompt}]}],
                               "generationConfig": {"temperature": 0.2, "responseMimeType": "application/json"}}, {})
        return out["candidates"][0]["content"]["parts"][0]["text"]


class OpenAICompatible(Provider):
    name = "openai"

    def _call(self, prompt: str) -> str:
        url = (self.base_url or "https://api.openai.com/v1").rstrip("/") + "/chat/completions"
        key = os.environ.get("OPENAI_API_KEY", "ollama")
        out = self._post(url, {"model": self.model, "temperature": 0.2,
                               "messages": [{"role": "user", "content": prompt}]},
                         {"Authorization": f"Bearer {key}"})
        return out["choices"][0]["message"]["content"]


PROVIDERS = {"gemini": Gemini, "openai": OpenAICompatible}


def parse_reply(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON in reply")
    d = json.loads(m.group(0))
    for k in ("hook", "cliffhanger", "standalone"):
        d[k] = max(1, min(5, int(d.get(k, 1))))
    return d


def validate(p: Promo, d: dict, ids: dict) -> dict:
    """Keep only what we can verify against the script."""
    notes = []
    corpus = " ".join(e.text for e in p.events)
    for k in ("hook_line", "cliffhanger_line"):
        q = (d.get(k) or "").strip().strip('"')
        if q and q[:40] not in corpus:
            notes.append(f"{k} not found verbatim in script - discarded")
            d[k] = None
    trim = None
    if len(p.segments) == 1 and (d.get("better_start") in ids or d.get("better_end") in ids):
        s = ids[d["better_start"]].start if d.get("better_start") in ids else p.segments[0].start
        e = ids[d["better_end"]].end if d.get("better_end") in ids else p.segments[0].end
        if MIN_LEN <= e - s <= MAX_LEN:
            trim = (s, e)
        else:
            notes.append("suggested trim breaks 20-60s rule - ignored")
    elif d.get("better_start") or d.get("better_end"):
        notes.append("suggested trim not usable - ignored")
    d["validated_trim"] = trim
    d["validation_notes"] = notes
    return d


def judge(promos: list[Promo], provider: Provider, premise: str, ctx=None, top_k: int = 60, log=print) -> list[Promo]:
    from .candidates import Segment, events_in, score_promo
    for p in promos:                       # unjudged clips keep their rules score
        p.scores["final"] = p.scores["rules"]
    for i, p in enumerate(sorted(promos, key=lambda x: x.score, reverse=True)[:top_k]):
        prompt, ids = build_prompt(p, premise)
        try:
            d = validate(p, parse_reply(provider.complete(prompt)), ids)
        except Exception as ex:
            p.flags.append(f"llm error: {type(ex).__name__}")
            continue
        if d["validated_trim"] and ctx is not None:
            ep = ctx.episodes[p.segments[0].episode]
            s, e = d["validated_trim"]
            p.segments = [Segment(ep.number, s, e)]
            p.events = events_in(ep, s, e)
            p.flags = ["trimmed on model suggestion (re-checked by rules)"]
            score_promo(p, ctx, at_episode_end=abs(e - ep.duration) < 0.5)
        p.llm = d
        llm_n = (0.4 * d["hook"] + 0.4 * d["cliffhanger"] + 0.2 * d["standalone"] - 1) / 4
        final = 0.5 * p.scores["rules"] + 0.5 * llm_n
        if d.get("reject"):
            final *= 0.7
            p.flags.append(f"llm rejected: {d.get('reject_reason', '')[:80]}")
        p.scores["llm"] = round(llm_n, 4)
        p.scores["final"] = round(final, 4)
        if (i + 1) % 10 == 0:
            log(f"  judged {i + 1}/{min(top_k, len(promos))}")
    return promos
