"""Stage 2: VAD/segmentation + loudness normalization.

Turns a raw clip (any format librosa/ffmpeg can decode, incl. .m4a) into:
  - normalized mono audio at a fixed sample rate
  - a list of speech segments (start/end in seconds)
  - a list of pauses (gaps between speech segments, incl. leading/trailing
    silence) — kept, not discarded, since pause structure is a prosodic
    feature (Stage 3 needs it for pause_count / pause_mean_dur / pause_var_dur).
"""

from dataclasses import dataclass

import librosa
import numpy as np
import webrtcvad

SAMPLE_RATE = 16000  # required by webrtcvad: 8000/16000/32000/48000
FRAME_MS = 30  # webrtcvad frame size: 10/20/30 ms
VAD_AGGRESSIVENESS = 2  # 0 (least aggressive) - 3 (most aggressive)
TARGET_RMS_DBFS = -20.0  # loudness normalization target
MAX_CLIP_DURATION_S = 15.0  # clips should be 3-10s per CLAUDE.md; guard against stray long recordings


@dataclass
class Segment:
    start: float  # seconds
    end: float  # seconds

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class PreprocessedClip:
    audio: np.ndarray
    sr: int
    speech_segments: list[Segment]
    pauses: list[Segment]


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load any librosa/ffmpeg-decodable audio file as mono float32 at `sr`."""
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def normalize_loudness(audio: np.ndarray, target_dbfs: float = TARGET_RMS_DBFS) -> np.ndarray:
    """Scale audio so its RMS level matches `target_dbfs`. No-op on silence."""
    rms = np.sqrt(np.mean(audio**2))
    if rms < 1e-9:
        return audio
    target_rms = 10 ** (target_dbfs / 20)
    gain = target_rms / rms
    normalized = audio * gain
    peak = np.max(np.abs(normalized))
    if peak > 1.0:
        normalized = normalized / peak
    return normalized.astype(np.float32)


def _frame_generator(audio_pcm16: bytes, sr: int, frame_ms: int):
    bytes_per_frame = int(sr * (frame_ms / 1000.0)) * 2  # 16-bit mono
    offset = 0
    while offset + bytes_per_frame <= len(audio_pcm16):
        yield audio_pcm16[offset : offset + bytes_per_frame]
        offset += bytes_per_frame


def detect_speech_segments(
    audio: np.ndarray,
    sr: int = SAMPLE_RATE,
    frame_ms: int = FRAME_MS,
    aggressiveness: int = VAD_AGGRESSIVENESS,
) -> list[Segment]:
    """Run webrtcvad over `audio` and merge consecutive speech frames into segments."""
    vad = webrtcvad.Vad(aggressiveness)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()

    frame_dur = frame_ms / 1000.0
    segments: list[Segment] = []
    seg_start = None
    for i, frame in enumerate(_frame_generator(pcm16, sr, frame_ms)):
        t = i * frame_dur
        is_speech = vad.is_speech(frame, sr)
        if is_speech and seg_start is None:
            seg_start = t
        elif not is_speech and seg_start is not None:
            segments.append(Segment(seg_start, t))
            seg_start = None
    if seg_start is not None:
        segments.append(Segment(seg_start, len(audio) / sr))
    return segments


def compute_pauses(speech_segments: list[Segment], clip_duration: float) -> list[Segment]:
    """Gaps between speech segments, including leading/trailing silence."""
    if not speech_segments:
        return [Segment(0.0, clip_duration)] if clip_duration > 0 else []

    pauses: list[Segment] = []
    if speech_segments[0].start > 0:
        pauses.append(Segment(0.0, speech_segments[0].start))
    for prev, nxt in zip(speech_segments, speech_segments[1:]):
        if nxt.start > prev.end:
            pauses.append(Segment(prev.end, nxt.start))
    if speech_segments[-1].end < clip_duration:
        pauses.append(Segment(speech_segments[-1].end, clip_duration))
    return pauses


def discover_clips(root: str, pattern: str = "*.m4a") -> list[str]:
    """Find clip paths under `root` (e.g. data/genuine/<speaker>/), skipping
    stray recordings well outside the expected 3-10s range."""
    import glob

    paths = sorted(glob.glob(f"{root}/**/{pattern}", recursive=True))
    kept = []
    for p in paths:
        dur = librosa.get_duration(path=p)
        if dur > MAX_CLIP_DURATION_S:
            print(f"skipping {p}: {dur:.1f}s exceeds MAX_CLIP_DURATION_S={MAX_CLIP_DURATION_S}s")
            continue
        kept.append(p)
    return kept


def preprocess(path: str, sr: int = SAMPLE_RATE) -> PreprocessedClip:
    """Load, loudness-normalize, and VAD-segment a clip."""
    audio = load_audio(path, sr=sr)
    audio = normalize_loudness(audio)
    speech_segments = detect_speech_segments(audio, sr=sr)
    pauses = compute_pauses(speech_segments, clip_duration=len(audio) / sr)
    return PreprocessedClip(audio=audio, sr=sr, speech_segments=speech_segments, pauses=pauses)


if __name__ == "__main__":
    import sys

    for clip_path in sys.argv[1:]:
        clip = preprocess(clip_path)
        n_speech = len(clip.speech_segments)
        n_pause = len(clip.pauses)
        speech_time = sum(s.duration for s in clip.speech_segments)
        print(
            f"{clip_path}: {len(clip.audio) / clip.sr:.2f}s total, "
            f"{n_speech} speech segment(s) ({speech_time:.2f}s), "
            f"{n_pause} pause(s)"
        )
