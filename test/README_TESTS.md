# Test Suite — Multimodal AI Inference API

## CHECK THE PORT IN CONFTEST.PY

Black-box HTTP tests for every endpoint documented in the API README:
`/media/info`, `/clip`, `/tinyclip`, `/clip4clip`, `/clap`, and `/av`
(search + label + batch_search where applicable).

## What's covered

For each endpoint:
- **Happy path** — valid input, correct response schema and types.
- **Ranking/consistency** — result lists are sorted by score where the
  README implies ranking; label scores form a softmax-like distribution;
  fused AV scores track the documented `fused = w_v*visual + w_a*audio`
  formula.
- **Parameter behavior** — `top_k` clamps rather than errors when it
  exceeds available results; `segment_seconds`/`max_frames` visibly
  change granularity; extreme `visual_weight`/`audio_weight` skew the
  fused score toward the dominant modality.
- **Negative/edge cases** — missing required fields, empty query/labels
  strings, corrupt or empty files, non-matching modality (e.g. audio sent
  to an image endpoint), a video with no audio track sent to `/av/search`,
  garbage `top_k` values, wrong HTTP method, unknown routes.
- **No 500s** — the hard rule throughout is that bad input must degrade to
  a clean 4xx, never an unhandled server error. Tests explicitly assert
  `status_code != 500` even where the "correct" 4xx code isn't documented,
  since that's the one behavior every case can be checked against without
  guessing internal validation details the README doesn't specify.

## Running

```bash
# 1. Start the API (from the project's own README)
python api.py

# 2. Install test deps and run, from this tests/ directory
pip install -r requirements-test.txt
pytest
```

By default the suite targets `http://localhost:8000`. Override with:

```bash
API_BASE_URL=http://some-host:8000 pytest
```

If the server isn't reachable, the whole session is **skipped** (not
failed) with a message telling you to start it — a skip here means "go
start the server," not "something is broken."

## Test media

Real model weights need real-ish input to produce meaningful
scores/rankings, but the suite must also run without the project's
`dataset/` and `demo_videos/` folders present. It resolves this two ways:

1. **Synthetic fixtures (default).** `conftest.py` uses `ffmpeg` (must be
   on `PATH`) to generate a small test image, a sine-wave audio clip, and
   a short video with both a visual test pattern and an audible tone —
   plus purpose-built bad inputs (corrupt bytes, empty files, a
   video with no audio track, text disguised as a `.jpg`).
2. **Real assets (recommended before a release).** Set `TEST_ASSETS_DIR`
   to the project's own `dataset/` folder (containing `images/` and
   `audio/`) and the suite will prefer those files, plus the sibling
   `demo_videos/sample.mp4`:

   ```bash
   TEST_ASSETS_DIR=/path/to/clips-demo/dataset pytest
   ```

## Known limitations

- **Semantic correctness isn't verified.** A sine tone and a test-pattern
  video can confirm an endpoint ranks/scores/shapes its response
  correctly, but can't confirm the model correctly identifies "a red
  sports car" in an image of an actual car — that needs the real
  `dataset/` assets (see above) and, ideally, a human or golden-answer
  check on top of this suite.
- **Exact 4xx status codes aren't asserted** for cases the README doesn't
  specify (e.g. whether a missing field is a 400 or a 422). Tests check
  "some 4xx, never a 500" in those spots rather than guessing a specific
  code and risking false failures on a reasonable implementation choice.
- **`/media/info`'s and `/clip4clip`'s server-local `*_path` input mode**
  is exercised lightly (existence/absence only) since it depends on
  paths being valid on the server's own filesystem, which a black-box
  client can't fully control.
- **Performance/load** isn't covered — this suite checks correctness, not
  throughput or latency under concurrency.
