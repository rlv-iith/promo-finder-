"""Parse a timestamped microdrama screenplay (.txt) into structured events.

Expected block shape (the format in the input pack):

    **LADY JULIANA (Confused, Breathless):**
    Start: (00:13.50) | End: (00:14.30)
    "Why?"

Labels CAMERA / ACTION / SOUND FX are non-dialogue; anything else is a speaker.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

TIME_RE = re.compile(r"(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)")
BLOCK_RE = re.compile(r"^\*\*(.+?):\*\*\s*$")
SPAN_RE = re.compile(r"^Start:\s*\(([^)]+)\)\s*\|\s*End:\s*\(([^)]+)\)")
NON_DIALOGUE = {"CAMERA": "camera", "ACTION": "action", "SOUND FX": "sfx", "SFX": "sfx"}


def to_seconds(ts: str) -> float:
    m = TIME_RE.search(ts)
    if not m:
        raise ValueError(f"bad timestamp: {ts!r}")
    h, mm, ss = m.groups()
    return (int(h) * 3600 if h else 0) + int(mm) * 60 + float(ss)


def fmt(t: float) -> str:
    """Seconds -> mm:ss.ss (same style as the scripts)."""
    m, s = divmod(round(t, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Event:
    episode: int
    scene: int
    idx: int            # order inside the episode
    kind: str           # dialogue | action | camera | sfx
    start: float
    end: float
    text: str
    speaker: str = ""   # dialogue only, normalised (e.g. "JULIANA")
    tags: list[str] = field(default_factory=list)  # e.g. ["Cold", "Commanding"]

    def to_dict(self):
        return asdict(self)


@dataclass
class Episode:
    number: int
    path: str
    logline: str
    continuity: str
    events: list[Event]
    scene_headings: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max((e.end for e in self.events), default=0.0)


# Name variants that refer to one character are merged so "LADY JULIANA" == "JULIANA".
TITLE_WORDS = {"LADY", "LORD", "THE", "SIR", "DUKE", "DUCHESS"}


def norm_speaker(raw: str) -> str:
    name = re.sub(r"\(.*?\)", "", raw).strip().upper()
    parts = name.split()
    # "LADY JULIANA" -> "JULIANA", but keep descriptive names like "LADY IN GOLD GOWN"
    if len(parts) == 2 and parts[0] in TITLE_WORDS:
        parts = parts[1:]
    return " ".join(parts)


def parse_episode(path: str | Path) -> Episode:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    header = {}
    for ln in lines[:10]:
        if ":" in ln and ln.split(":", 1)[0].isupper():
            k, v = ln.split(":", 1)
            header[k.strip()] = v.strip()
    num_m = re.search(r"\d+", header.get("EPISODE NUMBER", "")) or re.search(r"episode-(\d+)", path.name)
    number = int(num_m.group(1) if num_m.lastindex else num_m.group(0))

    events: list[Event] = []
    headings: dict[int, str] = {}
    scene = 0
    i = 0
    while i < len(lines):
        ln = lines[i].strip()
        if ln.startswith("SCENE "):
            scene = int(re.sub(r"\D", "", ln) or scene + 1)
        if ln.lower().startswith("scene heading:"):
            headings[scene] = ln.split(":", 1)[1].strip()
        m = BLOCK_RE.match(ln)
        if m and i + 1 < len(lines):
            span = SPAN_RE.match(lines[i + 1].strip())
            if span:
                label = m.group(1).strip()
                body = []
                j = i + 2
                while j < len(lines) and lines[j].strip() and not BLOCK_RE.match(lines[j].strip()) \
                        and not lines[j].startswith("---"):
                    body.append(lines[j].strip())
                    j += 1
                text = " ".join(body).strip()
                kind = NON_DIALOGUE.get(label.upper())
                tags = []
                speaker = ""
                if kind is None:
                    kind = "dialogue"
                    speaker = norm_speaker(label)
                    for grp in re.findall(r"\(([^)]*)\)", label):
                        if grp.upper() in {"O.S.", "V.O.", "CONT'D"}:
                            tags.append(grp.upper())
                        else:
                            tags += [t.strip() for t in grp.split(",") if t.strip()]
                    text = text.strip('"').strip("“”")
                events.append(Event(number, scene, len(events), kind,
                                    to_seconds(span.group(1)), to_seconds(span.group(2)),
                                    text, speaker, tags))
                i = j
                continue
        i += 1
    events.sort(key=lambda e: (e.start, e.idx))
    for k, e in enumerate(events):
        e.idx = k
    return Episode(number, str(path), header.get("EPISODE LOGLINE", ""),
                   header.get("CONTINUITY NOTES", ""), events, headings)


def load_show(folder: str | Path) -> list[Episode]:
    files = sorted(Path(folder).rglob("*.txt"))
    eps = [parse_episode(f) for f in files]
    eps = [e for e in eps if e.events]
    return sorted(eps, key=lambda e: e.number)
