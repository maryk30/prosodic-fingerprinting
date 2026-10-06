"""Shared protocol for the clone generators (generate_clones.py / XTTS and
generate_clones_f5.py / F5-TTS). No heavy imports: each generator runs in
its own virtualenv."""

import glob
import os
import re

SCRIPT = "docs/RECORDING_SCRIPT.md"
ENROLL_SENTENCES = range(1, 21)  # voice reference — the only audio a cloner sees
CLONE_SENTENCES = range(21, 31)  # text to synthesize — the held-out sentences


def sentences() -> dict[int, str]:
    with open(SCRIPT) as f:
        return {int(m.group(1)): m.group(2).strip() for m in re.finditer(r"^(\d+)\. (.+)$", f.read(), re.M)}


def enrollment_clips(speaker: str) -> list[str]:
    """The speaker's take-1 recordings of s01–s20 (zero padding optional),
    in sentence order."""
    out = []
    for n in ENROLL_SENTENCES:
        pat = re.compile(rf"^{re.escape(speaker)}_s0*{n}$")
        out += [
            p for p in glob.glob(f"data/genuine/{speaker}/*")
            if pat.match(os.path.splitext(os.path.basename(p))[0])
        ]
    return out
