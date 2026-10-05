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
reproduces how the speaker phrases that sentence.
"""

import glob
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


TD_SENTENCES = range(21, 31)  # the held-out / clone sentences in docs/RECORDING_SCRIPT.md
TD_QUANTILE = 0.9


def td_distance(m: dict) -> float:
    """Single text-dependent distance: melody/loudness shape + local timing.
    tempo_ratio is reported but not included — overall reading speed varies
    a lot between takes of the same speaker."""
    return m["contour_dist"] + m["timing_dist"]


def _all_takes(speaker: str, sentence: int, root: str = GENUINE_ROOT) -> list[str]:
    takes, k = [], 1
    while (p := find_take(speaker, sentence, k, root)) or k == 1:
        if p:
            takes.append(p)
        k += 1
    return takes


def genuine_pair_distances(speaker: str, exclude_sentence: int | None = None) -> list[float]:
    """Distances between take 1 and later takes of the same sentence by the
    same speaker — the spread of genuine same-sentence variation, used to
    calibrate the accept threshold. Empty until 2nd takes are recorded."""
    out = []
    for n in TD_SENTENCES:
        if n == exclude_sentence:
            continue
        takes = _all_takes(speaker, n)
        if len(takes) >= 2:
            ref = contour(takes[0])
            out += [td_distance(compare(ref, contour(t))) for t in takes[1:]]
    return out


def td_threshold(speaker: str, exclude_sentence: int | None = None) -> float | None:
    d = genuine_pair_distances(speaker, exclude_sentence)
    return float(np.quantile(d, TD_QUANTILE)) if len(d) >= 3 else None


def clone_takes(speaker: str, sentence: int) -> list[str]:
    """Synthetic clones of `speaker` reading `sentence`, from any TTS system
    (data/synthetic/<speaker>/<tts_system>/<speaker>_sNN.*)."""
    pattern = re.compile(TAKE_RE.format(speaker=re.escape(speaker), n=sentence))
    return sorted(
        p for p in glob.glob(os.path.join(SYNTHETIC_ROOT, speaker, "*", "*"))
        if pattern.match(os.path.splitext(os.path.basename(p))[0])
    )


def text_dependent_trials(speakers: list[str]) -> list[dict]:
    """For each target T and sentence n: reference = T's take 1 of n;
    genuine = T's later takes of n; impostor = every take of n by other
    humans; clone = synthetic clones of T reading n."""
    cache: dict[str, Contour] = {}

    def c(p: str) -> Contour:
        if p not in cache:
            cache[p] = contour(p)
        return cache[p]

    trials = []
    for target in speakers:
        for n in TD_SENTENCES:
            takes = _all_takes(target, n)
            if not takes:
                continue
            ref = c(takes[0])
            cands = (
                [(p, "genuine") for p in takes[1:]]
                + [(p, "impostor") for other in speakers if other != target for p in _all_takes(other, n)]
                + [(p, "clone") for p in clone_takes(target, n)]
            )
            for p, kind in cands:
                m = compare(ref, c(p))
                trials.append(
                    dict(target=target, sentence=n, clip=os.path.basename(p), kind=kind,
                         is_genuine=int(kind == "genuine"), td_distance=td_distance(m), **m)
                )
    return trials


if __name__ == "__main__":
    import sys
    import pandas as pd

    sys.path.insert(0, os.path.dirname(__file__))
    from eval import evaluate

    speakers = sorted(os.listdir(GENUINE_ROOT))
    speakers = [s for s in speakers if os.path.isdir(os.path.join(GENUINE_ROOT, s))]
    t = pd.DataFrame(text_dependent_trials(speakers))
    pd.set_option("display.float_format", "{:.3f}".format)
    metrics = ["contour_dist", "timing_dist", "tempo_ratio", "td_distance"]
    print(t["kind"].value_counts().to_string())
    print(t.groupby("kind")[metrics].mean().to_string())
    if "genuine" not in set(t["kind"]):
        print("\nNo genuine same-sentence pairs yet — record 2nd takes of s21–s30 as "
              "<speaker>_sNN_t2 (see docs/RECORDING_SCRIPT.md) for ROC-AUC/EER.")
    else:
        for negative in ["impostor", "clone"]:
            sub = t[t["kind"].isin(["genuine", negative])]
            if sub["kind"].nunique() < 2:
                continue
            print(f"\ngenuine vs {negative}:")
            for metric in metrics:
                m = evaluate(sub.is_genuine.to_numpy(), -sub[metric].to_numpy())
                print(f"  {metric:13s} ROC-AUC {m['roc_auc']:.3f}  EER {m['eer']:.3f}")
