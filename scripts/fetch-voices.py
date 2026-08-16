#!/usr/bin/env python3
"""One-time setup: fetch Bark speaker embeddings and build local .npz voices.

Needs network. Everything afterwards runs offline.

    python scripts/fetch-voices.py                 # 10 English voices
    python scripts/fetch-voices.py --voices en_speaker_0 en_speaker_9

Note: the Hugging Face Xet transfer backend has been observed to hang
partway through large downloads, so it is disabled here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # xet backend stalls on big files
os.environ.pop("HF_HUB_OFFLINE", None)             # this script needs network

BARK_HOME = Path.home() / ".bark"
RAW_DIR = BARK_HOME / "speaker_embeddings"
VOICE_DIR = BARK_HOME / "voices"
PARTS = ("semantic", "coarse", "fine")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--voices", nargs="*",
                    default=[f"en_speaker_{i}" for i in range(10)])
    ap.add_argument("--repo", default="suno/bark")
    args = ap.parse_args()

    try:
        import numpy as np
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        print(f"missing dependency: {exc}. pip install -r requirements.txt", file=sys.stderr)
        return 1

    VOICE_DIR.mkdir(parents=True, exist_ok=True)

    # generation_config.json is required because bark-small does not ship one.
    try:
        src = hf_hub_download(args.repo, filename="generation_config.json")
        (BARK_HOME / "generation_config.json").write_text(Path(src).read_text())
        print("ok  generation_config.json")
    except Exception as exc:
        print(f"WARN generation_config.json: {exc}", file=sys.stderr)

    built = 0
    for voice in args.voices:
        stem = voice.split("/")[-1]
        arrays = {}
        for part in PARTS:
            name = f"speaker_embeddings/{stem}_{part}_prompt.npy"
            try:
                path = hf_hub_download(args.repo, filename=name, local_dir=str(BARK_HOME))
                arrays[f"{part}_prompt"] = np.load(path)
            except Exception as exc:
                print(f"WARN {name}: {str(exc)[:70]}", file=sys.stderr)

        if len(arrays) == len(PARTS):
            np.savez(VOICE_DIR / f"{stem}.npz", **arrays)
            built += 1
            print(f"ok  {stem}.npz")
        else:
            print(f"SKIP {stem}: got {len(arrays)}/{len(PARTS)} parts", file=sys.stderr)

    print(f"\n{built}/{len(args.voices)} voices -> {VOICE_DIR}")
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
