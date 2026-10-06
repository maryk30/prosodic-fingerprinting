"""Stage 3c: text-dependent prosodic comparison.

The clip-level fingerprint (features.py) summarizes prosody with statistics
over the whole clip, so most of the *shape* of the intonation is lost and
sentence-to-sentence variation swamps speaker habits. When the candidate is
reading a sentence we already have the enrolled speaker saying (a known
prompt, as in challenge-response verification), we can compare the
contours directly:

  1. contour(): per-frame pitch (semitones vs the clip's own median, so no
     register information) and loudness (dB vs the clip's own median), over
     the voiced span, unvoiced gaps interpolated.
  2. compare(): align the two contours with DTW, then measure
       contour_dist  mean frame distance along the alignment path — does
                     the melody/loudness shape match?
       timing_dist   std of log local tempo ratio along the path — are the
                     same syllables stretched/compressed the same way?
                     (global speed difference is factored out)
       tempo_ratio   |log(duration ratio)| — overall speed difference

All three are register- and timbre-independent: a clone that perfectly
copies pitch level and voice colour gets no credit here unless it also
reproduces how a person phrases that sentence.

Naturalness check (no second takes needed). Comparing a candidate with the
claimed speaker's own earlier reading would need two genuine readings of
the same sentence to calibrate, which we don't have. Instead, every
enrolled speaker reads every prompt sentence, so a candidate claiming to be
speaker X is compared with the *other* enrolled speakers' readings of the
same sentence (a cohort): its naturalness distance is the mean
contour+timing distance to them. A real person reads a sentence within the
normal human spread; a clone that phrases it unnaturally lands further out.

  - calibrate on the enrollment sentences s01–s20: each speaker's reading
    vs the cohort gives the genuine spread; per-speaker threshold = its
    90th percentile (~10% genuine false-rejects by construction)
  - test on the held-out sentences s21–s30: genuine readings vs clones

This checks "is this read the way people read this sentence", not "is this
speaker X" — the clip-level fingerprint does the latter. With only two
cohort readers per sentence the distance is noisy; more enrolled speakers
make it steadier.
"""

import glob
import json
import os
import re
from dataclasses import dataclass

import librosa
import numpy as np

from features import _pitch_track
from preprocessing import preprocess

GENUINE_ROOT = "data/genuine"
SYNTHETIC_ROOT = "data/synthetic"
# Fixed channel scales so semitones and dB contribute comparably to the DTW
# frame distance (roughly one within-clip std of each); fixed, not
# per-clip z-normalization, so a speaker's habitual pitch *range* still counts.
PITCH_SCALE_ST = 3.0
ENERGY_SCALE_DB = 6.0
TEMPO_WINDOW = 10  # path steps (~160ms) over which local tempo is measured


@dataclass
class Contour:
    frames: np.ndarray  # (2, T): scaled pitch, scaled loudness
    duration_s: float


def contour(path: str) -> Contour:
    clip = preprocess(path)
    times, f0, voiced = _pitch_track(clip)
    idx = np.flatnonzero(voiced)
    if len(idx) < 10:
        raise ValueError(f"too little voiced speech in {path}")
    span = slice(idx[0], idx[-1] + 1)
    t, f, v = times[span], f0[span], voiced[span]

    st = 12 * np.log2(f[v] / np.median(f[v]))
    st = np.interp(t, t[v], st)  # bridge consonants/short gaps

    hop = int(round((times[1] - times[0]) * clip.sr))
    rms = librosa.feature.rms(y=clip.audio, frame_length=2 * hop, hop_length=hop)[0]
    db = 20 * np.log10(rms[: len(times)][span] + 1e-9)
    db = db - np.median(db)

    return Contour(
        frames=np.vstack([st / PITCH_SCALE_ST, db / ENERGY_SCALE_DB]),
        duration_s=float(t[-1] - t[0]),
    )


def compare(a: Contour, b: Contour) -> dict:
    D, wp = librosa.sequence.dtw(X=a.frames, Y=b.frames, metric="euclidean")
    wp = wp[::-1]  # librosa returns the path end-to-start
    contour_dist = float(D[-1, -1] / len(wp))

    ia, ib = wp[:, 0].astype(float), wp[:, 1].astype(float)
    k = TEMPO_WINDOW
    da, db = ia[k:] - ia[:-k], ib[k:] - ib[:-k]
    ok = (da > 0) & (db > 0)
    log_tempo = np.log(db[ok] / da[ok])
    global_log = np.log(b.frames.shape[1] / a.frames.shape[1])
    timing_dist = float(np.std(log_tempo - global_log)) if ok.sum() > 2 else np.nan

    return {
        "contour_dist": contour_dist,
        "timing_dist": timing_dist,
        "tempo_ratio": float(abs(np.log(b.duration_s / a.duration_s))),
    }


TAKE_RE = r"^{speaker}_s0*{n}(?:_t(\d+))?$"


def find_take(speaker: str, sentence: int, take: int = 1, root: str = GENUINE_ROOT) -> str | None:
    """Path of `<speaker>_sNN` (take 1) or `<speaker>_sNN_tK` (take K);
    zero-padding of NN is optional (krishiv_s1 and mary_s01 both match)."""
    pattern = re.compile(TAKE_RE.format(speaker=re.escape(speaker), n=sentence))
    for p in glob.glob(os.path.join(root, speaker, "*")):
        m = pattern.match(os.path.splitext(os.path.basename(p))[0])
        if m and int(m.group(1) or 1) == take:
            return p
    return None


ENROLL_SENTENCES = range(1, 21)  # calibration: every speaker read these
TEST_SENTENCES = range(21, 31)  # held out; also the text the clones speak
THRESHOLD_QUANTILE = 0.9
THRESHOLDS_JSON = "data/features/naturalness_thresholds.json"


def td_distance(m: dict) -> float:
    """Single text-dependent distance: melody/loudness shape + local timing.
    tempo_ratio is reported but not included — overall reading speed varies
    a lot between readings."""
    return m["contour_dist"] + m["timing_dist"]


def enrolled_speakers(root: str = GENUINE_ROOT) -> list[str]:
    return sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))


def clone_takes(speaker: str, sentence: int) -> list[tuple[str, str]]:
    """(tts_system, path) for clones of `speaker` reading `sentence`
    (data/synthetic/<speaker>/<tts_system>/<speaker>_sNN.*)."""
    pattern = re.compile(TAKE_RE.format(speaker=re.escape(speaker), n=sentence))
    return sorted(
        (p.split(os.sep)[-2], p)
        for p in glob.glob(os.path.join(SYNTHETIC_ROOT, speaker, "*", "*"))
        if pattern.match(os.path.splitext(os.path.basename(p))[0])
    )


class ContourCache:
    def __init__(self):
        self._c: dict[str, Contour] = {}

    def __call__(self, path: str) -> Contour:
        if path not in self._c:
            self._c[path] = contour(path)
        return self._c[path]


def naturalness(candidate: Contour, sentence: int, claimed: str, speakers: list[str], cache: ContourCache) -> float:
    """Mean contour+timing distance from the candidate to the other enrolled
    speakers' readings of the same sentence (the claimed speaker's own reading
    is never used, so genuine and clone trials are scored the same way)."""
    refs = [find_take(s, sentence) for s in speakers if s != claimed]
    refs = [r for r in refs if r]
    if not refs:
        raise ValueError(f"no cohort readings of sentence {sentence} besides {claimed}")
    return float(np.mean([td_distance(compare(cache(r), candidate)) for r in refs]))


def calibrate(speakers: list[str], cache: ContourCache) -> dict[str, float]:
    """Per-speaker accept threshold from their enrollment readings."""
    thresholds = {}
    for spk in speakers:
        d = [
            naturalness(cache(p), n, spk, speakers, cache)
            for n in ENROLL_SENTENCES
            if (p := find_take(spk, n))
        ]
        thresholds[spk] = float(np.quantile(d, THRESHOLD_QUANTILE))
    return thresholds


def test_trials(speakers: list[str], cache: ContourCache) -> list[dict]:
    trials = []
    for spk in speakers:
        for n in TEST_SENTENCES:
            cands = [("genuine", p) for p in [find_take(spk, n)] if p] + clone_takes(spk, n)
            for source, p in cands:
                try:
                    d = naturalness(cache(p), n, spk, speakers, cache)
                except ValueError:
                    continue
                trials.append(dict(speaker=spk, sentence=n, source=source,
                                   is_genuine=int(source == "genuine"), distance=d))
    return trials


def load_thresholds() -> dict[str, float]:
    with open(THRESHOLDS_JSON) as f:
        return json.load(f)


if __name__ == "__main__":
    import sys

    import pandas as pd

    sys.path.insert(0, os.path.dirname(__file__))
    from eval import evaluate

    speakers = enrolled_speakers()
    cache = ContourCache()
    thresholds = calibrate(speakers, cache)
    with open(THRESHOLDS_JSON, "w") as f:
        json.dump(thresholds, f, indent=2)
    print("thresholds (90th pct of enrollment naturalness distance):",
          {k: round(v, 3) for k, v in thresholds.items()}, "->", THRESHOLDS_JSON)

    t = pd.DataFrame(test_trials(speakers, cache))
    t["score"] = [thresholds[s] - d for s, d in zip(t["speaker"], t["distance"])]  # >0 = accept
    pd.set_option("display.float_format", "{:.3f}".format)
    print("\nmean naturalness distance, held-out s21-s30:")
    print(t.pivot_table(index="speaker", columns="source", values="distance").to_string())

    genuine = t[t["is_genuine"] == 1]
    rows = []
    for source in ["all"] + sorted(t.loc[t["is_genuine"] == 0, "source"].unique()):
        sub = t if source == "all" else pd.concat([genuine, t[t["source"] == source]])
        m = evaluate(sub["is_genuine"].to_numpy(), sub["score"].to_numpy())
        rows.append({"clones": source, "roc_auc": m["roc_auc"], "eer": m["eer"],
                     "genuine_accept": (sub.query("is_genuine == 1")["score"] >= 0).mean(),
                     "clone_reject": (sub.query("is_genuine == 0")["score"] < 0).mean()})
    print("\n" + pd.DataFrame(rows).set_index("clones").to_string())
