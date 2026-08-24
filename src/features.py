"""Stage 3: prosodic feature extraction.

Turns a preprocessed clip into the row schema defined in CLAUDE.md's data
contract:

    speaker, clip_id, label (genuine|synthetic), tts_system (na for genuine),
    f0_mean, f0_std, energy_mean, energy_std, speaking_rate_mean,
    pause_count, pause_mean_dur, pause_var_dur, npvi, jitter, shimmer

f0_mean/f0_std are in *semitones relative to the speaker's own median pitch*
(computed from their genuine enrollment clips), not raw Hz — this keeps the
fingerprint comparable across speakers with different pitch registers and
matches the "speaker-relative semitone normalization" requirement.
"""

import glob
import os
from dataclasses import dataclass

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call

from preprocessing import PreprocessedClip, discover_clips, preprocess

F0_MIN_HZ = 75
F0_MAX_HZ = 500
MIN_PAUSE_S = 0.05  # ignore sub-50ms gaps (VAD frame noise, not real pauses)
PAUSE_BOUNDARY_EPS = 0.02  # pauses touching clip start/end are silence, not speech pauses


@dataclass
class RawClipFeatures:
    speaker: str
    clip_id: str
    label: str
    tts_system: str
    voiced_f0_hz: np.ndarray  # for later speaker-relative semitone conversion
    energy_mean: float
    energy_std: float
    speaking_rate_mean: float
    pause_count: int
    pause_mean_dur: float
    pause_var_dur: float
    npvi: float
    jitter: float
    shimmer: float


FEATURE_COLUMNS = [
    "speaker",
    "clip_id",
    "label",
    "tts_system",
    "f0_mean",
    "f0_std",
    "energy_mean",
    "energy_std",
    "speaking_rate_mean",
    "pause_count",
    "pause_mean_dur",
    "pause_var_dur",
    "npvi",
    "jitter",
    "shimmer",
]


def _internal_pauses(clip: PreprocessedClip) -> list[float]:
    """Pause durations between speech segments, excluding leading/trailing
    silence (a recording artifact, not speaking behavior) and sub-frame noise."""
    clip_dur = len(clip.audio) / clip.sr
    durs = []
    for p in clip.pauses:
        if p.start <= PAUSE_BOUNDARY_EPS or p.end >= clip_dur - PAUSE_BOUNDARY_EPS:
            continue
        if p.duration < MIN_PAUSE_S:
            continue
        durs.append(p.duration)
    return durs


def _f0_track_hz(clip: PreprocessedClip) -> np.ndarray:
    f0, voiced_flag, _ = librosa.pyin(
        clip.audio, fmin=F0_MIN_HZ, fmax=F0_MAX_HZ, sr=clip.sr
    )
    return f0[voiced_flag]


def _energy_stats(clip: PreprocessedClip) -> tuple[float, float]:
    rms = librosa.feature.rms(y=clip.audio)[0]
    return float(np.mean(rms)), float(np.std(rms))


def _onset_times(clip: PreprocessedClip) -> np.ndarray:
    return librosa.onset.onset_detect(
        y=clip.audio, sr=clip.sr, units="time", backtrack=False
    )


def _speaking_rate(clip: PreprocessedClip, onsets: np.ndarray) -> float:
    speech_time = sum(s.duration for s in clip.speech_segments)
    if speech_time <= 0:
        return 0.0
    return len(onsets) / speech_time


def _npvi(onsets: np.ndarray) -> float:
    """Normalized Pairwise Variability Index over inter-onset intervals, as a
    rhythm-metric proxy for syllable-nucleus intervals."""
    if len(onsets) < 3:
        return float("nan")
    durs = np.diff(onsets)
    if len(durs) < 2:
        return float("nan")
    num = 0.0
    for d1, d2 in zip(durs, durs[1:]):
        denom = (d1 + d2) / 2
        if denom > 0:
            num += abs(d1 - d2) / denom
    return 100.0 * num / (len(durs) - 1)


def _jitter_shimmer(clip: PreprocessedClip) -> tuple[float, float]:
    try:
        sound = parselmouth.Sound(clip.audio, sampling_frequency=clip.sr)
        point_process = call(sound, "To PointProcess (periodic, cc)", F0_MIN_HZ, F0_MAX_HZ)
        jitter = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call(
            [sound, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6
        )
        return float(jitter), float(shimmer)
    except Exception:
        return float("nan"), float("nan")


def extract_raw_features(
    path: str, speaker: str, label: str, tts_system: str
) -> RawClipFeatures:
    clip = preprocess(path)
    onsets = _onset_times(clip)
    energy_mean, energy_std = _energy_stats(clip)
    pause_durs = _internal_pauses(clip)
    jitter, shimmer = _jitter_shimmer(clip)

    return RawClipFeatures(
        speaker=speaker,
        clip_id=os.path.splitext(os.path.basename(path))[0],
        label=label,
        tts_system=tts_system,
        voiced_f0_hz=_f0_track_hz(clip),
        energy_mean=energy_mean,
        energy_std=energy_std,
        speaking_rate_mean=_speaking_rate(clip, onsets),
        pause_count=len(pause_durs),
        pause_mean_dur=float(np.mean(pause_durs)) if pause_durs else 0.0,
        pause_var_dur=float(np.var(pause_durs)) if pause_durs else 0.0,
        npvi=_npvi(onsets),
        jitter=jitter,
        shimmer=shimmer,
    )


def speaker_f0_reference_hz(raw_clips: list[RawClipFeatures]) -> float:
    """Median voiced F0 (Hz) across a speaker's genuine clips — their own
    pitch register, used as the 0-semitone reference for all their clips
    (genuine and synthetic alike)."""
    genuine_f0 = np.concatenate(
        [r.voiced_f0_hz for r in raw_clips if r.label == "genuine" and len(r.voiced_f0_hz)]
    )
    return float(np.median(genuine_f0)) if len(genuine_f0) else float("nan")


def _speaker_f0_reference_hz(raw_by_speaker: dict[str, list[RawClipFeatures]]) -> dict[str, float]:
    return {speaker: speaker_f0_reference_hz(rows) for speaker, rows in raw_by_speaker.items()}


def to_row(r: RawClipFeatures, f0_ref_hz: float) -> dict:
    if len(r.voiced_f0_hz) and f0_ref_hz > 0:
        semitones = 12 * np.log2(r.voiced_f0_hz / f0_ref_hz)
        f0_mean, f0_std = float(np.mean(semitones)), float(np.std(semitones))
    else:
        f0_mean, f0_std = float("nan"), float("nan")

    return {
        "speaker": r.speaker,
        "clip_id": r.clip_id,
        "label": r.label,
        "tts_system": r.tts_system,
        "f0_mean": f0_mean,
        "f0_std": f0_std,
        "energy_mean": r.energy_mean,
        "energy_std": r.energy_std,
        "speaking_rate_mean": r.speaking_rate_mean,
        "pause_count": r.pause_count,
        "pause_mean_dur": r.pause_mean_dur,
        "pause_var_dur": r.pause_var_dur,
        "npvi": r.npvi,
        "jitter": r.jitter,
        "shimmer": r.shimmer,
    }


def build_features_table(
    genuine_root: str, synthetic_root: str | None = None
) -> tuple[list[dict], dict[str, float]]:
    raw_by_speaker: dict[str, list[RawClipFeatures]] = {}

    for speaker_dir in sorted(glob.glob(f"{genuine_root}/*/")):
        speaker = os.path.basename(speaker_dir.rstrip("/"))
        for clip_path in discover_clips(speaker_dir):
            raw = extract_raw_features(clip_path, speaker, label="genuine", tts_system="na")
            raw_by_speaker.setdefault(speaker, []).append(raw)

    if synthetic_root and os.path.isdir(synthetic_root):
        for speaker_dir in sorted(glob.glob(f"{synthetic_root}/*/")):
            speaker = os.path.basename(speaker_dir.rstrip("/"))
            for tts_dir in sorted(glob.glob(f"{speaker_dir}*/")):
                tts_system = os.path.basename(tts_dir.rstrip("/"))
                for clip_path in discover_clips(tts_dir):
                    raw = extract_raw_features(
                        clip_path, speaker, label="synthetic", tts_system=tts_system
                    )
                    raw_by_speaker.setdefault(speaker, []).append(raw)

    f0_refs = _speaker_f0_reference_hz(raw_by_speaker)
    rows = [
        to_row(r, f0_refs[speaker])
        for speaker, clips in raw_by_speaker.items()
        for r in clips
    ]
    return rows, f0_refs


F0_REFERENCE_PATH = "data/features/f0_reference.json"


if __name__ == "__main__":
    import json

    import pandas as pd

    rows, f0_refs = build_features_table("data/genuine", "data/synthetic")
    df = pd.DataFrame(rows, columns=FEATURE_COLUMNS)
    out_path = "data/features/features.csv"
    df.to_csv(out_path, index=False)
    print(f"wrote {len(df)} rows to {out_path}")
    print(df)

    with open(F0_REFERENCE_PATH, "w") as f:
        json.dump(f0_refs, f, indent=2)
    print(f"wrote per-speaker f0 reference (Hz) to {F0_REFERENCE_PATH}")
