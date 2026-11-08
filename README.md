# Promo Finder: finds promo clips in microdrama screenplays

Reads a show's timestamped episode scripts and returns ranked 20–60 s promo clips. For each clip you get the episode, the start and end of every segment, the hook (first 3 s), the cliffhanger, a one-line reason, and warnings where it may be weak.

- **Runs free and offline by default.** It uses Python's standard library only: no installs and no API key. A 50-episode show takes about 5 s.
- **Optional LLM judge.** Any model can be plugged in (Gemini free tier, Groq, OpenRouter, OpenAI, or a local Ollama). It scores and describes the shortlisted clips. It can't invent timestamps.

## Run it

```bash
python run.py --input "path/to/show-scripts.zip" --out outputs          # rules only
python -m unittest discover tests                                        # tests
python eval/evaluate.py outputs --final                                  # score vs. hand-labelled promos
```

To add the LLM judge (it only re-scores the top 60 candidates; replies are cached, so re-runs are free):

```bash
# Gemini free tier (https://aistudio.google.com/apikey)
export GEMINI_API_KEY=...            # Windows PowerShell: $env:GEMINI_API_KEY="..."
python run.py --input show.zip --out outputs_llm --provider gemini --model gemini-2.5-flash

# Any OpenAI-compatible API: Groq, OpenRouter, OpenAI, local Ollama...
export OPENAI_API_KEY=...
python run.py --input show.zip --out outputs_llm --provider openai \
  --base-url https://api.groq.com/openai/v1 --model llama-3.3-70b-versatile
```

Useful flags: `--per-episode 3`, `--min-score 0.60`, `--top 15`, `--no-stitch`, `--lexicon my_genre.json`.

## Outputs

| File | What it is |
|---|---|
| `report.md` | Readable report: the "post these first" list, then every promo by episode |
| `promos.csv` | One row per promo, for a spreadsheet or an editor's queue |
| `promos.json` | Everything: segments, scores per signal, the signals that fired, warnings, transcript |
| `eval_results.json` | Comparison with the hand-labelled reference promos in `eval/gold.json` |
| `manual_audit.md` | My editor-style review of the top 15 |
| `validation/` | Baselines, blind A/B judging and a human blind-test page. Start with `validation/VALIDATION.md` |

## Does it beat the obvious shortcuts?

| Check | Result |
|---|---|
| Match to 20 hand-labelled promos (strict IoU ≥ 0.5) | Tool 70%, random clips 52%, episode endings 40%. **Held-out half: 60%, which ties episode endings and is not clearly above random** |
| Blind A/B, tool's #1 vs the episode ending (35 pairs) | Tool preferred 74% (95% CI 58–86%, p = 0.003) |
| Blind A/B, tool's #1 vs a random clip (42 pairs) | Tool preferred 69% (95% CI 54–81%, p = 0.01) |

The blind judges are AI reviewers reading anonymised transcripts. A human version is ready to run:

```bash
python eval/baselines.py --input show.zip --tool outputs                    # baselines
python eval/blind_pairs.py make  --input show.zip --tool outputs            # blind pairs + blind_test.html
# people open outputs/validation/blind_test.html, vote, and press "Download votes"
python eval/blind_pairs.py score --votes votes_RK.json votes_AS.json        # win rate, 95% CI, p-value
```

## How it works

1. **Parse** (`parse.py`). The script becomes timed events: dialogue (with speaker and delivery tags such as *Furious, Mocking*), action, camera and sound effects.
2. **Cut points** (`candidates.py`). Clips may only start and end on a shot boundary and never inside a line of dialogue, so every clip can be cut as given.
3. **Score every 20–60 s window** using rules from a genre lexicon (`lexicon.json`):
   - *hook*: the loudest event in the first 3 s (visible violence, a reveal, a betrayal, a threat, a humiliation, heated delivery)
   - *cliffhanger*: the last line or beat leaves something open (a question, a cut-off line, a threat, a door bursting open). Endings that wrap up warmly lose points.
   - *standalone*: roles are spelled out (sister, husband, Duke), there are few unexplained names and no "remember when" lines, and the clip starts on a natural beat
   - *intensity*, *density* (little dead air), and whether the *lead characters* carry the scene
4. **Stitch** (`stitch.py`). It tries multi-segment promos, either cutting out a slow middle or joining the end of episode N to the start of N+1. A stitch is kept only if it beats the single clips it would replace by a clear margin.
5. **De-duplicate and rank** (`pipeline.py`). It keeps up to 3 non-overlapping promos per episode. The post-first order is quality, plus a nudge for early episodes (fast payoff when a viewer taps through to E1), plus a nudge when the clip shows what the title promises. It uses one promo per episode, to spread them out.
6. **Optional LLM judge** (`llm.py`). The model scores hook, cliffhanger and standalone from 1 to 5 and writes the reason. Code then discards any quote that isn't in the script word for word, and any trim that breaks the 20–60 s or cut rules.

## Adapting to a new show or genre

No code changes are needed: point `--input` at the new zip. For a different genre (werewolf, billionaire, revenge), copy `promo_finder/lexicon.json`, edit the word lists and pass `--lexicon`.

> Note: the input pack is confidential Kuku content. Don't commit the scripts to a public repo (they're excluded in `.gitignore`).
