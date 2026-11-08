"""Run: python -m unittest discover tests   (set SHOW_DIR to the episode folder)"""
import json
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from promo_finder.candidates import MAX_LEN, MIN_LEN, ShowContext  # noqa: E402
from promo_finder.llm import Provider, judge, parse_reply  # noqa: E402
from promo_finder.parse import load_show, parse_episode, to_seconds  # noqa: E402
from promo_finder.pipeline import generate  # noqa: E402

SAMPLE = """EPISODE NUMBER: Episode 7
EPISODE LOGLINE: A test.
CONTINUITY NOTES: None.

--- SCRIPT START ---

SCENE 01
Scene heading: INT. HALL – DAY
Scene start: 00:00.00
Scene end: 00:40.00

Actions and dialogue:

**CAMERA:**
Start: (00:00.00) | End: (00:05.00)
Close-up.

**ACTION:**
Start: (00:00.50) | End: (00:02.00)
She slaps him across the face.

**LADY JULIANA (Furious, Shouting):**
Start: (00:02.00) | End: (00:04.50)
"You betrayed me!"

**CAMERA:**
Start: (00:05.00) | End: (00:40.00)
Wide.

**GERALD (Cold):**
Start: (00:30.00) | End: (00:33.00)
"Who told you?"
--- SCRIPT END ---"""


class FakeLLM(Provider):
    """Returns a scripted reply - lets us test validation without an API key."""
    name = "fake"

    def __init__(self, reply, **kw):
        super().__init__(cache_dir=kw.get("cache_dir", "/tmp/pf_test_cache"), min_interval=0)
        self.reply = reply

    def complete(self, prompt):
        return self.reply


class ParseTests(unittest.TestCase):
    def setUp(self):
        self.p = Path("/tmp/pf_sample_ep.txt")
        self.p.write_text(SAMPLE, encoding="utf-8")

    def test_parse(self):
        ep = parse_episode(self.p)
        self.assertEqual(ep.number, 7)
        dl = [e for e in ep.events if e.kind == "dialogue"]
        self.assertEqual(dl[0].speaker, "JULIANA")          # "LADY JULIANA" -> JULIANA
        self.assertEqual(dl[0].tags, ["Furious", "Shouting"])
        self.assertEqual(dl[0].text, "You betrayed me!")
        self.assertAlmostEqual(ep.duration, 40.0)
        self.assertEqual(to_seconds("01:02.50"), 62.5)

    def test_llm_validation_rejects_invented_quotes_and_bad_trims(self):
        ep = parse_episode(self.p)
        ctx = ShowContext([ep])
        promos = generate([ep], ctx, stitch=False)
        self.assertTrue(promos)
        reply = json.dumps({"hook": 5, "cliffhanger": 9, "standalone": 4,
                            "hook_line": "I will destroy your whole family tonight",   # not in script
                            "cliffhanger_line": "Who told you?",
                            "why": "test", "better_start": "L99", "better_end": None, "reject": False})
        judge(promos, FakeLLM(reply), "premise", ctx=ctx, top_k=1)
        d = promos[0].llm
        self.assertIsNone(d["hook_line"])                     # hallucinated quote discarded
        self.assertEqual(d["cliffhanger_line"], "Who told you?")
        self.assertEqual(d["cliffhanger"], 5)                 # clamped to 1..5
        self.assertIsNone(d["validated_trim"])                # unknown line id ignored

    def test_parse_reply_handles_prose_wrapping(self):
        d = parse_reply('Sure! ```json\n{"hook": 3, "cliffhanger": 2, "standalone": 4}\n```')
        self.assertEqual((d["hook"], d["cliffhanger"]), (3, 2))


@unittest.skipUnless(os.environ.get("SHOW_DIR"), "set SHOW_DIR to run on the real input pack")
class ShowTests(unittest.TestCase):
    def test_every_promo_obeys_the_rules(self):
        eps = load_show(os.environ["SHOW_DIR"])
        ctx = ShowContext(eps)
        promos = generate(eps, ctx)
        by_ep = {e.number: e for e in eps}
        for p in promos:
            self.assertTrue(MIN_LEN - 0.01 <= p.duration <= MAX_LEN + 0.01, p.key)
            for s in p.segments:
                for d in by_ep[s.episode].events:
                    if d.kind != "dialogue":
                        continue
                    # never cut through a line of dialogue
                    self.assertFalse(d.start + 0.05 < s.start < d.end - 0.05, (p.key, d.text))
                    self.assertFalse(d.start + 0.05 < s.end < d.end - 0.05, (p.key, d.text))


if __name__ == "__main__":
    unittest.main()
