"""
Local Bark TTS web service.

Paste a script -> get audio. Runs entirely on this machine; the process
never reaches the network (HF_HUB_OFFLINE is forced before transformers
is imported).

Design constraints that shape this file:

  * Bark is autoregressive over an audio codec and truncates past roughly
    13s of audio per generate() call, so script-length input must be split
    on sentence boundaries and stitched back together.
  * Generation runs 9-20x slower than realtime, far too slow to answer an
    HTTP request inline. Work goes on a queue; the client polls.
  * One worker thread only. A second concurrent generation just makes both
    slower on a CPU-bound model and doubles peak RSS.
  * Model weights are 1.6GB (small) / 4.5GB (big). Holding both on a 16GB
    machine caused an out-of-memory kill in practice, so at most
    MAX_RESIDENT_MODELS stay loaded and the rest are evicted.
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")       # must precede transformers
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gc
import json
import logging
import re
import threading
import time
import uuid
import wave
from collections import OrderedDict
from contextlib import asynccontextmanager
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, Path as PathParam
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger("bark")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# --------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------

BARK_HOME = Path.home() / ".bark"
VOICE_DIR = BARK_HOME / "voices"
GEN_JSON = BARK_HOME / "generation_config.json"
OUT_DIR = BARK_HOME / "output"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS: Dict[str, str] = {"small": "suno/bark-small", "big": "suno/bark"}
DEFAULT_MODEL = "small"

# Bark degrades and truncates past roughly this much text per generate() call.
MAX_CHUNK_CHARS = 200
# Silence inserted between chunks so sentences do not butt together.
GAP_SECONDS = 0.25
# Refuse scripts beyond this; at ~9x realtime a 20k-char script is ~3 hours.
MAX_TEXT_CHARS = 20_000
# A single chunk that has not returned in this long is treated as hung.
CHUNK_TIMEOUT_S = 900
# Retained finished jobs (and their wav files). Oldest are evicted.
MAX_JOBS = 50
# Loaded model variants held in RAM at once.
MAX_RESIDENT_MODELS = 1

_JOB_ID_RE = re.compile(r"^[0-9a-f]{12}$")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class ModelKey(str, Enum):
    small = "small"
    big = "big"


class JobStatus(str, Enum):
    queued = "queued"
    running = "running"
    done = "done"
    error = "error"


# --------------------------------------------------------------------------
# text chunking
# --------------------------------------------------------------------------

def split_script(text: str, limit: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split a script into generation-sized chunks on sentence boundaries.

    Guarantees, both covered by tests:
      * no returned chunk exceeds ``limit`` characters
      * concatenating the result reproduces the input up to whitespace

    Sentences longer than ``limit`` are split again on commas, then on hard
    word boundaries, so an over-long sentence still cannot produce a chunk
    that Bark would silently truncate.
    """
    text = text.strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]

    pieces: List[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            pieces.append(sentence)
            continue
        buf = ""
        for part in re.split(r"(?<=,)\s+", sentence):
            if len(buf) + len(part) + 1 <= limit:
                buf = f"{buf} {part}".strip()
            else:
                if buf:
                    pieces.append(buf)
                buf = part
        if buf:
            pieces.append(buf)

    wrapped: List[str] = []
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            if cut <= 0:
                cut = limit
            wrapped.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            wrapped.append(piece)

    packed: List[str] = []
    for piece in wrapped:
        if packed and len(packed[-1]) + len(piece) + 1 <= limit:
            packed[-1] = f"{packed[-1]} {piece}"
        else:
            packed.append(piece)
    return packed


# --------------------------------------------------------------------------
# wire contracts
# --------------------------------------------------------------------------

class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice: Optional[str] = Field(None, max_length=128)
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    model: ModelKey = ModelKey(DEFAULT_MODEL)

    @field_validator("text")
    @classmethod
    def _speakable(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("text is empty or whitespace only")
        return v

    @field_validator("voice")
    @classmethod
    def _safe_voice(cls, v: Optional[str]) -> Optional[str]:
        # Names reach the filesystem. resolve_voice() also confines the result
        # to VOICE_DIR, but reject obvious traversal here as well so a bad
        # value never propagates past the edge.
        if not v:
            return v
        if not re.fullmatch(r"(?:v\d+/)?[A-Za-z0-9_-]+", v):
            raise ValueError(
                "voice must be a bare name, optionally prefixed 'v2/' "
                "(letters, digits, underscore, hyphen)"
            )
        return v


class JobView(BaseModel):
    id: str
    status: JobStatus
    progress: float
    done_chunks: int
    total_chunks: int
    duration: float
    elapsed: float
    error: Optional[str] = None
    voice: Optional[str] = None
    model: ModelKey


class SpeakAccepted(JobView):
    eta_seconds: int


class ModelInfo(BaseModel):
    id: str
    available: bool
    loaded: bool


class Health(BaseModel):
    ready: bool
    error: Optional[str]
    sample_rate: int
    voices: List[str]
    models: Dict[str, ModelInfo]
    queued: int
    offline: bool
    worker_alive: bool
    worker_restarts: int
    jobs_retained: int


# --------------------------------------------------------------------------
# job state
# --------------------------------------------------------------------------

class Job:
    __slots__ = ("id", "text", "voice", "temperature", "model", "status",
                 "total_chunks", "done_chunks", "error", "duration",
                 "elapsed", "created")

    def __init__(self, id: str, text: str, voice: Optional[str],
                 temperature: float, model: ModelKey, total_chunks: int):
        self.id = id
        self.text = text
        self.voice = voice
        self.temperature = temperature
        self.model = model
        self.status = JobStatus.queued
        self.total_chunks = total_chunks
        self.done_chunks = 0
        self.error: Optional[str] = None
        self.duration = 0.0
        self.elapsed = 0.0
        self.created = time.time()

    def view(self) -> JobView:
        pct = (self.done_chunks / self.total_chunks * 100) if self.total_chunks else 0.0
        return JobView(
            id=self.id, status=self.status, progress=round(pct, 1),
            done_chunks=self.done_chunks, total_chunks=self.total_chunks,
            duration=round(self.duration, 2), elapsed=round(self.elapsed, 1),
            error=self.error, voice=self.voice, model=self.model,
        )


JOBS: "OrderedDict[str, Job]" = OrderedDict()
QUEUE: "Queue[str]" = Queue()
_LOCK = threading.RLock()

_ENGINES: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()
_ENGINE_ERR: Optional[str] = None
_SR = 24000
_WORKER_RESTARTS = 0
_WORKER: Optional[threading.Thread] = None
_SHUTDOWN = threading.Event()


def _prune_jobs_locked() -> None:
    """Drop oldest finished jobs (and their audio) past MAX_JOBS."""
    while len(JOBS) > MAX_JOBS:
        for jid, job in list(JOBS.items()):
            if job.status in (JobStatus.done, JobStatus.error):
                JOBS.pop(jid, None)
                try:
                    (OUT_DIR / f"{jid}.wav").unlink(missing_ok=True)
                except OSError as exc:
                    log.warning("could not remove audio for %s: %s", jid, exc)
                break
        else:
            break            # only unfinished jobs remain; keep them


# --------------------------------------------------------------------------
# engine
# --------------------------------------------------------------------------

def list_voices() -> List[str]:
    if not VOICE_DIR.is_dir():
        return []
    return sorted(p.stem for p in VOICE_DIR.glob("*.npz"))


def resolve_voice(name: Optional[str]) -> Optional[str]:
    """Map a voice name to a local .npz, refusing anything outside VOICE_DIR."""
    if not name:
        return None
    stem = Path(name).name                       # strips any directory part
    if not stem or stem.startswith("."):
        return None
    path = (VOICE_DIR / f"{stem}.npz").resolve()
    if not path.is_relative_to(VOICE_DIR.resolve()) or not path.exists():
        return None
    return str(path)


def model_available(key: str) -> bool:
    """True if the variant's weights are on disk (no network needed)."""
    repo = MODELS[key].replace("/", "--")
    root = Path.home() / ".cache/huggingface/hub" / f"models--{repo}" / "snapshots"
    if not root.is_dir():
        return False
    return any(
        (snap / "pytorch_model.bin").exists() or (snap / "model.safetensors").exists()
        for snap in root.iterdir() if snap.is_dir()
    )


def load_engine(key: str) -> Dict[str, Any]:
    """Load a Bark variant, evicting others to stay within MAX_RESIDENT_MODELS.

    Worker thread only; not safe to call concurrently.
    """
    global _ENGINE_ERR, _SR

    if key in _ENGINES:
        _ENGINES.move_to_end(key)
        return _ENGINES[key]

    import torch
    from transformers import AutoModel, AutoProcessor, GenerationConfig

    while len(_ENGINES) >= MAX_RESIDENT_MODELS:
        victim, _ = _ENGINES.popitem(last=False)
        log.info("evicting model '%s' to free memory", victim)
        gc.collect()

    model_id = MODELS[key]
    try:
        proc = AutoProcessor.from_pretrained(model_id)
        model = AutoModel.from_pretrained(model_id).eval()

        # bark-small ships no generation_config.json; the nested schema is
        # shared across bark variants so a cached copy is a valid stand-in.
        if not hasattr(model.generation_config, "semantic_config"):
            if not GEN_JSON.exists():
                raise RuntimeError(
                    f"{model_id} has no semantic_config and {GEN_JSON} is missing"
                )
            model.generation_config = GenerationConfig(**json.loads(GEN_JSON.read_text()))

        torch.set_num_threads(max(1, (os.cpu_count() or 4) - 1))

        _SR = int(model.generation_config.sample_rate)
        _ENGINES[key] = {"model": model, "proc": proc, "sr": _SR}
        _ENGINE_ERR = None
        log.info("loaded %s (resident: %s)", model_id, list(_ENGINES))
        return _ENGINES[key]
    except Exception as exc:
        _ENGINE_ERR = f"{model_id}: {type(exc).__name__}: {exc}"
        log.error("engine load failed: %s", _ENGINE_ERR)
        raise


def generate_chunk(text: str, voice_path: Optional[str], temperature: float,
                   eng: Dict[str, Any]) -> np.ndarray:
    import torch

    proc, model = eng["proc"], eng["model"]
    inputs = (proc(text=text, voice_preset=voice_path, return_tensors="pt")
              if voice_path else proc(text=text, return_tensors="pt"))
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=True, temperature=temperature)

    audio = out[0].cpu().numpy().astype(np.float32)
    if audio.size == 0:
        raise RuntimeError("model returned empty audio")
    if not np.all(np.isfinite(audio)):
        raise RuntimeError("model returned non-finite samples")
    return audio


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    peak = float(np.abs(audio).max()) if audio.size else 0.0
    if peak > 0:
        audio = audio / peak * 0.95
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).astype("<i2")
    tmp = path.with_suffix(".wav.part")           # never serve a half-written file
    with wave.open(str(tmp), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(sr)
        fh.writeframes(pcm.tobytes())
    tmp.replace(path)


def run_job(job: Job) -> None:
    started = time.time()
    job.status = JobStatus.running

    chunks = split_script(job.text)
    job.total_chunks = len(chunks)
    if not chunks:
        raise ValueError("no speakable text")

    eng = load_engine(job.model.value)
    voice_path = resolve_voice(job.voice)
    if job.voice and voice_path is None:
        log.warning("job %s: voice %r unavailable, using model default",
                    job.id, job.voice)

    sr = eng["sr"]
    gap = np.zeros(int(sr * GAP_SECONDS), dtype=np.float32)

    parts: List[np.ndarray] = []
    for idx, chunk in enumerate(chunks):
        t0 = time.time()
        parts.append(generate_chunk(chunk, voice_path, job.temperature, eng))
        if time.time() - t0 > CHUNK_TIMEOUT_S:
            raise TimeoutError(
                f"chunk {idx} took {time.time()-t0:.0f}s (limit {CHUNK_TIMEOUT_S}s)"
            )
        if idx != len(chunks) - 1:
            parts.append(gap)
        job.done_chunks = idx + 1
        job.elapsed = time.time() - started

    audio = np.concatenate(parts)
    write_wav(OUT_DIR / f"{job.id}.wav", audio, sr)
    job.duration = len(audio) / sr
    job.elapsed = time.time() - started
    job.status = JobStatus.done
    log.info("job %s done: %.2fs audio in %.1fs", job.id, job.duration, job.elapsed)


def worker_loop() -> None:
    """Drain the queue. Exceptions fail the job, never the thread."""
    try:
        load_engine(DEFAULT_MODEL)               # warm the default up front
    except Exception:
        log.exception("initial engine warmup failed; will retry per job")

    while not _SHUTDOWN.is_set():
        try:
            job_id = QUEUE.get(timeout=1.0)
        except Empty:
            continue

        job = JOBS.get(job_id)
        if job is None:
            QUEUE.task_done()
            continue

        try:
            run_job(job)
        except Exception as exc:
            job.status = JobStatus.error
            job.error = f"{type(exc).__name__}: {exc}"
            log.exception("job %s failed", job_id)
        finally:
            QUEUE.task_done()


def supervise() -> None:
    """Restart the worker if it ever dies, so the queue never stalls silently."""
    global _WORKER, _WORKER_RESTARTS
    while not _SHUTDOWN.is_set():
        if _WORKER is None or not _WORKER.is_alive():
            if _WORKER is not None:
                _WORKER_RESTARTS += 1
                log.error("worker died; restart #%d", _WORKER_RESTARTS)
            _WORKER = threading.Thread(target=worker_loop, daemon=True,
                                       name="bark-worker")
            _WORKER.start()
        _SHUTDOWN.wait(5.0)


# --------------------------------------------------------------------------
# api
# --------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    threading.Thread(target=supervise, daemon=True, name="bark-supervisor").start()
    yield
    _SHUTDOWN.set()


app = FastAPI(title="Bark TTS (local)", version="1.0.0", lifespan=lifespan)


def _get_job(job_id: str) -> Job:
    if not _JOB_ID_RE.fullmatch(job_id):
        raise HTTPException(400, "malformed job id")
    job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(404, "unknown job")
    return job


@app.get("/api/health", response_model=Health)
def health() -> Health:
    return Health(
        ready=bool(_ENGINES),
        error=_ENGINE_ERR,
        sample_rate=_SR,
        voices=list_voices(),
        models={k: ModelInfo(id=v, available=model_available(k), loaded=k in _ENGINES)
                for k, v in MODELS.items()},
        queued=QUEUE.qsize(),
        offline=os.environ.get("HF_HUB_OFFLINE") == "1",
        worker_alive=bool(_WORKER and _WORKER.is_alive()),
        worker_restarts=_WORKER_RESTARTS,
        jobs_retained=len(JOBS),
    )


@app.post("/api/speak", response_model=SpeakAccepted, status_code=202)
def speak(req: SpeakRequest) -> SpeakAccepted:
    chunks = split_script(req.text)
    if not chunks:
        raise HTTPException(422, "no speakable text")
    if not model_available(req.model.value):
        raise HTTPException(
            409, f"model '{req.model.value}' ({MODELS[req.model.value]}) is not downloaded")

    job = Job(uuid.uuid4().hex[:12], req.text, req.voice,
              req.temperature, req.model, len(chunks))
    with _LOCK:
        JOBS[job.id] = job
        _prune_jobs_locked()
    QUEUE.put(job.id)

    rate = 9 if req.model == ModelKey.small else 20   # x realtime, measured
    return SpeakAccepted(**job.view().model_dump(),
                         eta_seconds=int(len(chunks) * 11 * rate))


@app.get("/api/jobs", response_model=List[JobView])
def list_jobs() -> List[JobView]:
    with _LOCK:
        return [j.view() for j in reversed(JOBS.values())]


@app.get("/api/jobs/{job_id}", response_model=JobView)
def job_status(job_id: str = PathParam(..., min_length=12, max_length=12)) -> JobView:
    return _get_job(job_id).view()


@app.delete("/api/jobs/{job_id}", status_code=204)
def delete_job(job_id: str = PathParam(..., min_length=12, max_length=12)) -> Response:
    job = _get_job(job_id)
    with _LOCK:
        JOBS.pop(job.id, None)
    (OUT_DIR / f"{job.id}.wav").unlink(missing_ok=True)
    return Response(status_code=204)


@app.get("/api/jobs/{job_id}/audio")
def job_audio(job_id: str = PathParam(..., min_length=12, max_length=12)) -> Response:
    job = _get_job(job_id)
    if job.status is not JobStatus.done:
        raise HTTPException(409, f"job is {job.status.value}")

    path = OUT_DIR / f"{job_id}.wav"
    if not path.exists():
        raise HTTPException(410, "audio no longer available")
    return Response(
        path.read_bytes(),
        media_type="audio/wav",
        headers={"Content-Disposition": f'inline; filename="bark_{job_id}.wav"'},
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    page = BARK_HOME / "index.html"
    if not page.exists():
        raise HTTPException(500, "index.html missing")
    return page.read_text()
