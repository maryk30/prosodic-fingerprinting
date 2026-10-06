"""Stage 3: prosodic feature extraction.

Turns a preprocessed clip into the row schema defined in CLAUDE.md's data
contract:

    speaker, clip_id, label (genuine|synthetic), tts_system (na for genuine),
    f0_mean, f0_std, energy_mean, energy_std, speaking_rate_mean,
    pause_count, pause_mean_dur, pause_var_dur, npvi, jitter, shimmer

Stage 3b added register-independent prosodic columns after shimmer:

    f0_range, f0_slope, f0_velocity, f0_final_move, energy_slope,
    voiced_fraction, varco_v, varco_uv

These are computed in semitones/dB relative to the *clip's own* median, so
they describe how pitch and loudness move, not where the voice sits — i.e.
habit, not register or timbre.

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
PYIN_FRAME = 1024  # 64ms at 16kHz: >= 2 periods at F0_MIN_HZ
PYIN_HOP = 256  # 16ms: fine enough to time voiced/unvoiced stretches (syllable ~100-250ms)
FINAL_WINDOW_S = 0.3  # last 300ms of voicing = the utterance's boundary tone
MIN_VOICED_RUN_FRAMES = 2  # shorter voiced blips are pYIN noise, not vowels
MAX_UNVOICED_RUN_S = 0.3  # longer unvoiced stretches inside speech are pauses, not consonants


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
    f0_range: float
    f0_slope: float
    f0_velocity: float
    f0_final_move: float
    energy_slope: float
    voiced_fraction: float
    varco_v: float
    varco_uv: float


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
    "f0_range",
    "f0_slope",
    "f0_velocity",
    "f0_final_move",
    "energy_slope",
    "voiced_fraction",
    "varco_v",
    "varco_uv",
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


def _pitch_track(clip: PreprocessedClip) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(frame times s, f0 Hz with NaN where unvoiced, voiced flag)."""
    f0, voiced_flag, _ = librosa.pyin(
        clip.audio,
        fmin=F0_MIN_HZ,
        fmax=F0_MAX_HZ,
        sr=clip.sr,
        frame_length=PYIN_FRAME,
        hop_length=PYIN_HOP,
    )
    times = librosa.times_like(f0, sr=clip.sr, hop_length=PYIN_HOP)
    return times, f0, voiced_flag.astype(bool)


def _runs(flags: np.ndarray) -> list[tuple[bool, int]]:
    """Run-length encode a boolean array: [(value, length), ...]."""
    runs = []
    for v in flags:
        if runs and runs[-1][0] == v:
            runs[-1] = (v, runs[-1][1] + 1)
        else:
            runs.append((bool(v), 1))
    return runs


def _pitch_dynamics(times: np.ndarray, f0: np.ndarray, voiced: np.ndarray) -> dict:
    """How pitch *moves*, in semitones relative to the clip's own median —
    independent of the speaker's register:
      f0_range       p90-p10 spread (st)
      f0_slope       linear trend over the utterance (st/s; declination)
      f0_velocity    mean |pitch change| between adjacent voiced frames (st/s)
      f0_final_move  last 300ms of voicing vs clip median (st; boundary rise/fall)
    """
    nan = dict(f0_range=np.nan, f0_slope=np.nan, f0_velocity=np.nan, f0_final_move=np.nan)
    if voiced.sum() < 10:
        return nan
    st = np.full_like(f0, np.nan)
    st[voiced] = 12 * np.log2(f0[voiced] / np.median(f0[voiced]))
    t, v = times[voiced], st[voiced]
    hop_s = times[1] - times[0]
    adjacent = voiced[1:] & voiced[:-1]
    return dict(
        f0_range=float(np.percentile(v, 90) - np.percentile(v, 10)),
        f0_slope=float(np.polyfit(t, v, 1)[0]),
        f0_velocity=float(np.mean(np.abs(np.diff(st)[adjacent])) / hop_s) if adjacent.any() else np.nan,
        f0_final_move=float(np.mean(v[t >= t[-1] - FINAL_WINDOW_S])),
    )


def _rhythm(times: np.ndarray, voiced: np.ndarray) -> dict:
    """Voiced/unvoiced interval rhythm metrics, a forced-alignment-free proxy
    for Ramus/Dellwo vocalic/consonantal metrics:
      voiced_fraction  %V-like: voiced time / (voiced + short unvoiced) time
      varco_v          100 * std/mean of voiced-run durations
      varco_uv         100 * std/mean of within-speech unvoiced-run durations
    Unvoiced runs longer than MAX_UNVOICED_RUN_S are pauses, so excluded.
    """
    nan = dict(voiced_fraction=np.nan, varco_v=np.nan, varco_uv=np.nan)
    idx = np.flatnonzero(voiced)
    if len(idx) < 10:
        return nan
    hop_s = times[1] - times[0]
    runs = _runs(voiced[idx[0] : idx[-1] + 1])  # trim leading/trailing silence
    v_durs = np.array([n * hop_s for val, n in runs if val and n >= MIN_VOICED_RUN_FRAMES])
    uv_durs = np.array([n * hop_s for val, n in runs if not val and n * hop_s <= MAX_UNVOICED_RUN_S])

    def varco(d: np.ndarray) -> float:
        return float(100 * np.std(d) / np.mean(d)) if len(d) >= 3 else np.nan

    total = v_durs.sum() + uv_durs.sum()
    return dict(
        voiced_fraction=float(v_durs.sum() / total) if total > 0 else np.nan,
        varco_v=varco(v_durs),
        varco_uv=varco(uv_durs),
    )


def _energy_slope(clip: PreprocessedClip) -> float:
    """Loudness trend over speech frames (dB/s) — does the speaker trail off?"""
    hop = 256
    rms = librosa.feature.rms(y=clip.audio, frame_length=512, hop_length=hop)[0]
    t = librosa.times_like(rms, sr=clip.sr, hop_length=hop)
    in_speech = speech_mask(t, clip)
    if in_speech.sum() < 10:
        return np.nan
    db = 20 * np.log10(rms[in_speech] + 1e-9)
    return float(np.polyfit(t[in_speech], db, 1)[0])


def speech_mask(times: np.ndarray, clip: PreprocessedClip) -> np.ndarray:
    """True for frame times inside a VAD speech segment.

    Every prosodic and voice-quality feature is measured on speech only.
    Measuring over the whole clip let non-speech sound leak in: F5-TTS clones
    carry ~0.6 s of low hum before speech starts in 28/30 clips, which pYIN
    tracks as a flat "voiced" pitch and which then faked a rising pitch slope
    and slow pitch movement (Stage 10 finding, 2026-10-06)."""
    mask = np.zeros(len(times), dtype=bool)
    for seg in clip.speech_segments:
        mask |= (times >= seg.start) & (times < seg.end)
    return mask


def _speech_only_audio(clip: PreprocessedClip) -> np.ndarray:
    """Audio with everything outside speech segments zeroed (Praat finds no
    periods in silence, so jitter/shimmer then come from speech alone)."""
    t = np.arange(len(clip.audio)) / clip.sr
    return np.where(speech_mask(t, clip), clip.audio, 0.0)


def _energy_stats(clip: PreprocessedClip) -> tuple[float, float]:
    rms = librosa.feature.rms(y=clip.audio)[0]
    t = librosa.times_like(rms, sr=clip.sr)
    rms = rms[speech_mask(t, clip)] if speech_mask(t, clip).any() else rms
    return float(np.mean(rms)), float(np.std(rms))


def _onset_times(clip: PreprocessedClip) -> np.ndarray:
    onsets = librosa.onset.onset_detect(
        y=clip.audio, sr=clip.sr, units="time", backtrack=False
    )
    return onsets[speech_mask(onsets, clip)]


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
        sound = parselmouth.Sound(_speech_only_audio(clip), sampling_frequency=clip.sr)
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
    times, f0, voiced = _pitch_track(clip)
    voiced = voiced & speech_mask(times, clip)

    return RawClipFeatures(
        speaker=speaker,
        clip_id=os.path.splitext(os.path.basename(path))[0],
        label=label,
        tts_system=tts_system,
        voiced_f0_hz=f0[voiced],
        energy_mean=energy_mean,
        energy_std=energy_std,
        speaking_rate_mean=_speaking_rate(clip, onsets),
        pause_count=len(pause_durs),
        pause_mean_dur=float(np.mean(pause_durs)) if pause_durs else 0.0,
        pause_var_dur=float(np.var(pause_durs)) if pause_durs else 0.0,
        npvi=_npvi(onsets),
        jitter=jitter,
        shimmer=shimmer,
        energy_slope=_energy_slope(clip),
        **_pitch_dynamics(times, f0, voiced),
        **_rhythm(times, voiced),
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
        "f0_range": r.f0_range,
        "f0_slope": r.f0_slope,
        "f0_velocity": r.f0_velocity,
        "f0_final_move": r.f0_final_move,
        "energy_slope": r.energy_slope,
        "voiced_fraction": r.voiced_fraction,
        "varco_v": r.varco_v,
        "varco_uv": r.varco_uv,
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
