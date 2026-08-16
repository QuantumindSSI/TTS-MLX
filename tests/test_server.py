"""Tests for the Bark TTS service.

Covers the two chunker invariants the audio quality depends on, the
validation boundaries, and the path-traversal defences. No model is
loaded, so the whole suite runs in well under a second.
"""

import os
import re
import sys

import pytest

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "barktts"))

from server import (                                          # noqa: E402
    MAX_CHUNK_CHARS, MAX_TEXT_CHARS, ModelKey, SpeakRequest,
    resolve_voice, split_script,
)

_ws = lambda s: re.sub(r"\s+", "", s)


# ---------------------------------------------------------------- chunker

SCRIPTS = [
    "One. Two. Three.",
    "Welcome to the demonstration. This sentence is short.\n"
    "This one is considerably longer, containing several clauses, each "
    "separated by commas, so that the splitter has to fall back to comma "
    "boundaries rather than sentence boundaries alone.\nShort again! Yes.",
    "No punctuation at all just a long run of words " * 12,
    "Hyphenated-words and 3.14 decimals and Dr. Smith abbreviations.",
    "A" * 1000,                                    # single unbroken token
    "Ends without terminal punctuation",
    "Multiple!!! Punctuation??? Marks...",
    "Line one.\n\nLine two.\n\nLine three.",
    "café naïve résumé — em dashes and ünicode",
]


@pytest.mark.parametrize("text", SCRIPTS)
def test_no_chunk_exceeds_limit(text):
    """Oversized chunks get silently truncated by Bark -> lost audio."""
    for chunk in split_script(text):
        assert len(chunk) <= MAX_CHUNK_CHARS, f"{len(chunk)} > {MAX_CHUNK_CHARS}"


@pytest.mark.parametrize("text", SCRIPTS)
def test_chunking_is_lossless(text):
    """Every character must survive the round trip, ignoring whitespace."""
    assert _ws("".join(split_script(text))) == _ws(text)


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t"])
def test_empty_input_yields_no_chunks(text):
    assert split_script(text) == []


def test_no_empty_chunks():
    for text in SCRIPTS:
        assert all(c.strip() for c in split_script(text))


def test_short_sentences_are_packed():
    """Packing keeps generate() call count down; one call per word would crawl."""
    assert len(split_script("A. B. C. D. E. F. G.")) == 1


# ------------------------------------------------------------- validation

def test_rejects_text_over_cap():
    with pytest.raises(ValueError):
        SpeakRequest(text="x" * (MAX_TEXT_CHARS + 1))


def test_rejects_blank_text():
    with pytest.raises(ValueError):
        SpeakRequest(text="   \n  ")


@pytest.mark.parametrize("temp", [0.05, 1.6, -1.0])
def test_rejects_temperature_out_of_range(temp):
    with pytest.raises(ValueError):
        SpeakRequest(text="hi", temperature=temp)


@pytest.mark.parametrize("temp", [0.1, 0.7, 1.5])
def test_accepts_temperature_in_range(temp):
    assert SpeakRequest(text="hi", temperature=temp).temperature == temp


def test_rejects_unknown_model():
    with pytest.raises(ValueError):
        SpeakRequest(text="hi", model="enormous")


def test_defaults():
    r = SpeakRequest(text="hi")
    assert r.model is ModelKey.small and r.voice is None and r.temperature == 0.7


@pytest.mark.parametrize("voice", ["../../etc/passwd", "a;rm -rf /", "x\x00y", "a b"])
def test_rejects_illegal_voice_characters(voice):
    with pytest.raises(ValueError):
        SpeakRequest(text="hi", voice=voice)


# -------------------------------------------------------- path traversal

@pytest.mark.parametrize("voice", [
    "../../../etc/passwd",
    "/etc/passwd",
    "..",
    ".hidden",
    "does_not_exist",
])
def test_resolve_voice_refuses_escape(voice):
    """Must never resolve to a path outside VOICE_DIR."""
    assert resolve_voice(voice) is None


def test_resolve_voice_accepts_known(tmp_path, monkeypatch):
    import server
    monkeypatch.setattr(server, "VOICE_DIR", tmp_path)
    (tmp_path / "en_speaker_0.npz").write_bytes(b"stub")
    assert resolve_voice("en_speaker_0") is not None
    assert resolve_voice("v2/en_speaker_0") is not None      # prefix stripped


def test_resolve_voice_none_passthrough():
    assert resolve_voice(None) is None
