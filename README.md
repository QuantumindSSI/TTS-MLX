# TTS-MLX

Local neural text-to-speech on macOS. Paste a script into a browser, get a
WAV back. After a one-time model download, the service runs entirely
offline — verified with Wi-Fi switched off.

Built on [Suno Bark](https://huggingface.co/suno/bark) via PyTorch +
Hugging Face Transformers.

> **On the name.** This repository is called `TTS-MLX`, but the code here
> does **not** use Apple MLX. Inference runs on PyTorch. There is no MLX
> weight conversion in this repo. See
> [MLX status](#mlx-status-not-implemented) for exactly where that stands
> and why. The name reflects an intended direction, not the current
> implementation.

---

## What it does

- Paste a script of any length; it is split on sentence boundaries and stitched
  back into one continuous WAV
- 10 English speaker voices, consistent across chunks
- Two model sizes: `bark-small` (~400M params) and `bark` (~1.1B)
- Job queue with live progress — the browser never blocks on a long generation
- Fully offline: `HF_HUB_OFFLINE=1` is set before `transformers` is imported,
  so the process cannot silently reach the network

## Requirements

- macOS on Apple Silicon (works on Intel/Linux, just slower)
- Python 3.10+
- ~6GB disk for both model variants
- 8GB RAM minimum

## Install

```bash
git clone https://github.com/QuantumindSSI/TTS-MLX.git
cd TTS-MLX
pip install -r requirements.txt

# one-time, needs network; downloads voices + generation config
python scripts/fetch-voices.py

# one-time, needs network; downloads model weights (~1.6GB or ~4.5GB)
HF_HUB_DISABLE_XET=1 python -c "
from transformers import AutoModel, AutoProcessor
AutoProcessor.from_pretrained('suno/bark-small')
AutoModel.from_pretrained('suno/bark-small')"
```

`HF_HUB_DISABLE_XET=1` is not optional. See [Known issues](#known-issues).

## Run

```bash
./scripts/bark-web              # http://127.0.0.1:8080
./scripts/bark-web 9000         # different port
./scripts/bark-web 8080 --lan   # reachable from other devices
```

`BARK_PYTHON` overrides the interpreter if `python3` is not the right one.

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/health` | readiness, voices, model availability, worker state |
| `POST` | `/api/speak` | submit a script → `202` + job id |
| `GET` | `/api/jobs` | list retained jobs |
| `GET` | `/api/jobs/{id}` | status and progress |
| `GET` | `/api/jobs/{id}/audio` | the WAV |
| `DELETE` | `/api/jobs/{id}` | drop a job and its audio |

```bash
curl -X POST localhost:8080/api/speak \
  -H 'Content-Type: application/json' \
  -d '{"text":"Hello there.","voice":"en_speaker_6","model":"small"}'
# {"id":"a1b2c3d4e5f6","status":"queued",...,"eta_seconds":99}

curl localhost:8080/api/jobs/a1b2c3d4e5f6
curl localhost:8080/api/jobs/a1b2c3d4e5f6/audio -o out.wav
```

## Performance

Bark is autoregressive over an audio codec. It is **slow, and that is
inherent** — not a configuration problem.

Measured on an M-series Mac, 16GB:

| Model | Speed | 200-char script | 1,000 chars |
|---|---|---|---|
| `bark-small` | 9–13x slower than realtime | ~2 min | ~8 min |
| `bark` | ~20x slower than realtime | ~4 min | ~18 min |

The UI shows a live estimate as you type. Jobs survive tab closure; poll
`/api/jobs/{id}` to collect later.

## Architecture

```
browser ──POST /api/speak──> FastAPI ──> Queue ──> worker thread
   │                            │                      │
   └──GET /api/jobs/{id}────────┘                 Bark (PyTorch)
                                                       │
                                              ~/.bark/output/{id}.wav
```

One worker thread by design: Bark is CPU-bound, so a second concurrent
generation makes both slower and doubles peak memory.

### Chunking

Bark truncates past roughly 13s of audio per `generate()` call, so longer
input must be split. `split_script()` splits on sentence boundaries, falls
back to commas, then to hard word wrapping. Two invariants are enforced by
tests:

1. no chunk exceeds `MAX_CHUNK_CHARS` (an oversized chunk is silently
   truncated by the model — audio would be lost)
2. concatenating the chunks reproduces the input up to whitespace

The same voice `.npz` is reused for every chunk so the speaker stays
consistent, and chunks are joined with 0.25s of silence.

### Resource limits

| Guard | Value | Reason |
|---|---|---|
| `MAX_TEXT_CHARS` | 20,000 | at ~9x realtime, more is a multi-hour job |
| `MAX_RESIDENT_MODELS` | 1 | holding both variants OOM-killed a 16GB machine |
| `MAX_JOBS` | 50 | bounds the job table and the WAV directory |
| `CHUNK_TIMEOUT_S` | 900 | a hung generation would block the queue forever |

The worker runs under a supervisor thread that restarts it if it dies;
`/api/health` exposes `worker_alive` and `worker_restarts`.

## Tests

```bash
pip install pytest
python -m pytest tests/ -q      # 44 tests, <1s, loads no model
```

Covers chunker invariants (including unicode, unbroken 1000-char tokens,
and missing terminal punctuation), request validation boundaries, and
path-traversal defences on voice names.

## Security

- Binds to `127.0.0.1` by default; `--lan` is opt-in and warns
- Voice names are validated at the edge and re-confined to the voice
  directory before any filesystem access
- Job ids are matched against `^[0-9a-f]{12}$` before touching a path
- Requesting a model whose weights are absent returns `409` rather than
  triggering a download

There is **no authentication**. Do not expose this to an untrusted network.

## MLX status (not implemented)

Apple MLX would be the natural way to accelerate this on Apple Silicon.
It is not done, and this repo contains no MLX code.

What was attempted and abandoned:

- A hand-written MLX reimplementation of Bark's transformer stack was
  started, then discarded. It was architecturally incomplete — no
  multi-head attention splitting, no `lm_head` projections, wrong
  causality on the fine acoustics model, and **no EnCodec decoder**, which
  is the component that actually turns codec tokens into a waveform. It
  never loaded a real weight and never produced audio.
- The work was also premised on a misdiagnosis: PyTorch generation
  appeared to hang, which was attributed to compute cost. The real cause
  was the Hugging Face Xet transfer backend stalling mid-download. Once
  that was fixed, PyTorch inference worked immediately.

What a real MLX port would require:

1. Port three transformer stacks (semantic, coarse, fine) with correct
   multi-head attention and the fine model's non-causal attention
2. Port the EnCodec decoder — convolutional stack, LSTM, and residual
   vector quantiser
3. Convert `pytorch_model.bin` to MLX arrays with a verified name mapping
4. Validate numerically against the PyTorch reference, not just by ear

Contributions welcome. Until then, this is a PyTorch application.

## Known issues

**Xet backend hangs.** `hf-xet` was observed stalling at exactly
67,074,363 bytes on every attempt at a 4.5GB download — the identical byte
offset each time, which rules out a network flake. Set
`HF_HUB_DISABLE_XET=1` for any large download; with it disabled the same
file completed at ~9MB/s.

**`bark-small` ships no `generation_config.json`.** Its nested
`semantic_config` / `coarse_acoustics_config` / `fine_acoustics_config`
schema is required by `BarkModel.generate()`. `scripts/fetch-voices.py`
fetches the config from the full `suno/bark` repo, where the schema is
identical.

## Licence

MIT. Bark itself is MIT-licensed by Suno; the model weights carry their own
terms.
