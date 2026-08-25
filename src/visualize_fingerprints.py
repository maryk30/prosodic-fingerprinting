"""Renders each speaker's voice fingerprint (spectral + prosodic) as PNGs:
a mel-spectrogram of a representative clip, an MFCC coefficient profile,
and a cross-speaker prosodic feature comparison. Used to build the
panel-facing fingerprint presentation — not part of the detection pipeline
itself.
"""

import glob
import json
import os

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from preprocessing import discover_clips, preprocess

PANEL_BG = "#0D1220"
PANEL_FG = "#ECEEF5"
PANEL_ACCENT = "#4FC3CE"
PANEL_ACCENT2 = "#E9B369"
PANEL_GRID = "#2A3350"

OUT_DIR = "data/features/fingerprint_visuals"


def _panel_style(fig, ax):
    fig.patch.set_facecolor(PANEL_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=PANEL_FG, labelsize=9)
    for spine in ax.spines.values():
        spine.set_color(PANEL_GRID)
    ax.xaxis.label.set_color(PANEL_FG)
    ax.yaxis.label.set_color(PANEL_FG)
    ax.title.set_color(PANEL_FG)


def _representative_clip(speaker_dir: str) -> str:
    """The clip closest to this speaker's median duration — a typical
    example rather than a cherry-picked longest/shortest one."""
    clips = discover_clips(speaker_dir)
    durations = [librosa.get_duration(path=c) for c in clips]
    median = np.median(durations)
    return clips[int(np.argmin(np.abs(np.array(durations) - median)))]


def plot_spectrogram(speaker: str, genuine_root: str = "data/genuine") -> str:
    clip_path = _representative_clip(f"{genuine_root}/{speaker}")
    clip = preprocess(clip_path)
    mel = librosa.feature.melspectrogram(y=clip.audio, sr=clip.sr, n_mels=80)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    img = librosa.display.specshow(
        mel_db, sr=clip.sr, x_axis="time", y_axis="mel", ax=ax, cmap="mako_r"
        if "mako_r" in plt.colormaps() else "viridis",
    )
    _panel_style(fig, ax)
    ax.set_title(f"{speaker} — mel spectrogram (representative clip)", fontsize=10)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mel frequency")
    cbar = fig.colorbar(img, ax=ax, format="%+2.0f dB")
    cbar.ax.yaxis.set_tick_params(color=PANEL_FG, labelsize=8)
    plt.setp(cbar.ax.get_yticklabels(), color=PANEL_FG)
    cbar.outline.set_edgecolor(PANEL_GRID)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{speaker}_spectrogram.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=PANEL_BG)
    plt.close(fig)
    return out_path


def plot_mfcc_profile(speaker: str, fingerprint_dir: str = "data/features/fingerprints") -> str:
    with open(f"{fingerprint_dir}/{speaker}_spectral.json") as f:
        fp = json.load(f)

    coeffs = list(range(1, 14))
    means = [fp[f"mfcc{i}_mean_mean"] for i in coeffs]
    stds = [np.sqrt(fp[f"mfcc{i}_mean_var"]) for i in coeffs]

    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.bar(coeffs, means, yerr=stds, color=PANEL_ACCENT, ecolor=PANEL_GRID, capsize=3, width=0.65)
    ax.axhline(0, color=PANEL_GRID, linewidth=0.8)
    _panel_style(fig, ax)
    ax.set_title(f"{speaker} — MFCC coefficient profile", fontsize=10)
    ax.set_xlabel("MFCC coefficient")
    ax.set_ylabel("value (mean ± std across clips)")
    ax.set_xticks(coeffs)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/{speaker}_mfcc_profile.png"
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, facecolor=PANEL_BG)
    plt.close(fig)
    return out_path


def plot_prosodic_comparison(
    speakers: list[str], features_csv: str = "data/features/features.csv"
) -> str:
    """Small multiples, one subplot per feature in its own real units — with
    only two speakers, normalizing to [0,1] across just those two points
    trivially forces one to 0 and the other to 1 regardless of how close
    the actual values are, hiding magnitude. Real units avoid that."""
    df = pd.read_csv(features_csv)
    genuine = df[df["label"] == "genuine"]

    specs = [
        ("f0_std", "pitch variability\n(semitones)"),
        ("speaking_rate_mean", "speaking rate\n(onsets/s)"),
        ("npvi", "rhythm\n(nPVI)"),
        ("jitter", "jitter"),
        ("shimmer", "shimmer"),
    ]
    colors = [PANEL_ACCENT, PANEL_ACCENT2]

    fig, axes = plt.subplots(1, len(specs), figsize=(9.6, 3.2))
    for ax, (col, label) in zip(axes, specs):
        vals = [genuine[genuine["speaker"] == s][col].mean() for s in speakers]
        ax.bar(speakers, vals, color=colors[: len(speakers)], width=0.55)
        _panel_style(fig, ax)
        ax.set_title(label, fontsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.margins(y=0.15)

    fig.suptitle("Prosodic profile — krishiv vs. mary (each in its own units)", fontsize=10, color=PANEL_FG)

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = f"{OUT_DIR}/prosodic_comparison.png"
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=180, facecolor=PANEL_BG)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    speakers = sorted(os.path.basename(d.rstrip("/")) for d in glob.glob("data/genuine/*/"))
    for speaker in speakers:
        print("spectrogram:", plot_spectrogram(speaker))
        print("mfcc profile:", plot_mfcc_profile(speaker))
    print("prosodic comparison:", plot_prosodic_comparison(speakers))
