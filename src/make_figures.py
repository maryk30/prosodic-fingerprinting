"""Figures for the paper (docs/paper/figures/*.pdf), from the same data the
pipeline and VoiceGuard Lab use.

    python src/export_viz.py && python src/make_figures.py

fig-contours     pitch contour of one sentence: genuine vs XTTS-v2 vs F5-TTS
fig-roc          ROC curves on held-out clones, per cloner
fig-tells        per-feature genuine-vs-clone separation, XTTS-v2 vs F5-TTS
fig-naturalness  naturalness distance by source, with the accept thresholds
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from fingerprint import PROSODIC_FEATURES, REGISTER_FEATURES, VOICE_QUALITY_FEATURES  # noqa: E402
from impostor_eval import HELDOUT_FIRST_SENTENCE, SYSTEMS, clone_trials, load_table  # noqa: E402

OUT = "docs/paper/figures"
COL = {"genuine": "#2a78d6", "xtts": "#eb6834", "f5tts": "#1baf7a"}
NAME = {"genuine": "Genuine", "xtts": "XTTS-v2", "f5tts": "F5-TTS"}
INK, INK2, MUTED, RULE = "#15181d", "#444b55", "#6b7380", "#d9dde2"

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Latin Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "legend.frameon": False, "pdf.fonttype": 42,
})


def lab_data() -> dict:
    with open("web/lab_data.js") as f:
        s = f.read()
    return json.loads(s[s.index("=") + 1: s.rindex(";")])


def fig_contours(D: dict, speaker="mary", sentence=25) -> None:
    clips = {c["source"]: c for c in D["clips"] if c["speaker"] == speaker and c["sentence"] == sentence}
    fig, axes = plt.subplots(3, 1, figsize=(6.3, 4.4), sharex=True, sharey=True)
    tmax = max(c["t"][-1] for c in clips.values())
    for ax, src in zip(axes, ["genuine", "xtts", "f5tts"]):
        c = clips[src]
        for a, b in c["speech"]:
            ax.axvspan(a, min(b, tmax), color="#eef2f7", lw=0)
        t = np.array(c["t"]); p = np.array([np.nan if v is None else v for v in c["pitch"]])
        ax.plot(t, p, color=COL[src], lw=1.6, solid_capstyle="round")
        ax.axhline(0, color=RULE, lw=0.6, zorder=0)
        ax.text(1.01, 0.5, NAME[src], transform=ax.transAxes, va="center", color=INK, fontsize=9)
        ax.set_ylabel("st")
        ax.grid(axis="y", color=RULE, lw=0.4, ls=(0, (2, 3)))
    axes[-1].set_xlabel("time (s)")
    axes[0].set_xlim(0, tmax)
    fig.suptitle(f"“{D['sentences'][str(sentence)]}” ({speaker}, s{sentence})", fontsize=9, color=INK2, y=0.995)
    fig.tight_layout(rect=(0, 0, 0.9, 0.97))
    fig.savefig(f"{OUT}/fig-contours.pdf"); plt.close(fig)


def fig_roc(D: dict) -> None:
    df, refs = load_table(include_synthetic=True)
    systems = {"Weighted blend": SYSTEMS["blend (default, capped)"], "Prosody only": SYSTEMS["prosody only"]}
    trials = {k: clone_trials(df, refs, v) for k, v in systems.items()}
    nat = pd.DataFrame([{"source": c["source"], "is_genuine": int(c["source"] == "genuine"),
                         "score": c["naturalness"]["threshold"] - c["naturalness"]["distance"]} for c in D["clips"]])
    style = {"Weighted blend": dict(color=INK, ls="-"), "Prosody only": dict(color=INK2, ls="--"),
             "Naturalness check": dict(color=MUTED, ls=":")}
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 3.0), sharey=True)
    for ax, tts in zip(axes, ["f5tts", "xtts"]):
        curves = {}
        for k, t in trials.items():
            sub = pd.concat([t[t.is_genuine == 1], t[t.source == tts]])
            curves[k] = (sub.is_genuine.to_numpy(), sub.score.to_numpy())
        sub = nat[nat.source.isin(["genuine", tts])]
        curves["Naturalness check"] = (sub.is_genuine.to_numpy(), sub.score.to_numpy())
        for k, (y, s) in curves.items():
            fpr, tpr, _ = roc_curve(y, s)
            ax.plot(fpr, tpr, lw=1.5, label=f"{k} ({roc_auc_score(y, s):.3f})", **style[k])
        ax.plot([0, 1], [0, 1], color=RULE, lw=0.6)
        ax.set_title(f"{NAME[tts]} clones", fontsize=9, color=INK)
        ax.set_xlabel("false accept rate (clones accepted)")
        ax.set_aspect("equal"); ax.set_xlim(-0.01, 1); ax.set_ylim(0, 1.01)
        ax.legend(loc="lower right", fontsize=7.5, handlelength=2.2, title="ROC-AUC", title_fontsize=7.5)
    axes[0].set_ylabel("genuine accept rate")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig-roc.pdf"); plt.close(fig)


LABELS = {
    "f0_velocity": "Pitch velocity", "shimmer": "Shimmer", "f0_slope": "Pitch slope", "energy_std": "Loudness variability",
    "jitter": "Jitter", "f0_range": "Pitch range", "speaking_rate_mean": "Speaking rate", "f0_mean": "Pitch level",
    "f0_std": "Pitch variability", "f0_final_move": "Final rise/fall", "pause_count": "Pause count",
    "pause_var_dur": "Pause variability", "pause_mean_dur": "Pause length", "voiced_fraction": "Voiced fraction",
    "energy_slope": "Loudness slope", "varco_v": "Vowel-length variation", "varco_uv": "Consonant-length variation",
    "npvi": "Rhythm (nPVI)",
}


def separation_table() -> pd.DataFrame:
    df, _ = load_table(include_synthetic=True)
    g = df[(df.label == "genuine") & (df.sentence >= HELDOUT_FIRST_SENTENCE)]
    rows = []
    for f in REGISTER_FEATURES + PROSODIC_FEATURES + VOICE_QUALITY_FEATURES:
        gz = g.groupby("speaker")[f].transform(lambda x: (x - x.mean()) / max(x.std(), 1e-9))
        r = {"feature": f}
        for tts in ["xtts", "f5tts"]:
            c = df[(df.label == "synthetic") & (df.tts_system == tts)]
            cz = pd.concat([(c[c.speaker == s][f] - g[g.speaker == s][f].mean()) / max(g[g.speaker == s][f].std(), 1e-9)
                            for s in c.speaker.unique()])
            m, mc = gz.notna(), cz.notna()
            a = roc_auc_score([1] * m.sum() + [0] * mc.sum(), list(gz[m]) + list(cz[mc]))
            r[tts] = max(a, 1 - a)
        rows.append(r)
    return pd.DataFrame(rows)


def fig_tells() -> None:
    t = separation_table()
    t["best"] = t[["xtts", "f5tts"]].max(axis=1)
    t = t.sort_values("best")
    fig, ax = plt.subplots(figsize=(6.3, 4.3))
    y = np.arange(len(t))
    for i, (_, r) in enumerate(t.iterrows()):
        ax.plot([r.xtts, r.f5tts], [i, i], color=RULE, lw=1.2, zorder=1)
    for tts, mk in [("xtts", "o"), ("f5tts", "D")]:
        ax.scatter(t[tts], y, s=26, color=COL[tts], marker=mk, zorder=2, edgecolor="white", linewidth=0.8, label=f"{NAME[tts]} clones")
    ax.set_yticks(y, [LABELS[f] for f in t.feature], color=INK2)
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 3)))
    ax.set_xlim(0.45, 1.02)
    ax.set_xlabel("separation from genuine speech (single-feature ROC-AUC; 0.5 = none)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", color=RULE, lw=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig-tells.pdf"); plt.close(fig)
    t.drop(columns="best").to_csv(f"{OUT}/separation.csv", index=False)


def fig_naturalness(D: dict) -> None:
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(6.3, 2.3))
    thr = sorted({c["naturalness"]["threshold"] for c in D["clips"]})
    ax.axvspan(min(thr), max(thr), color="#e9eef3", lw=0, zorder=0)
    ax.annotate(f"accept thresholds ({min(thr):.2f}–{max(thr):.2f})", xy=(max(thr), 2.45), xytext=(max(thr) + 0.45, 2.45),
                fontsize=7.5, color=INK2, va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.set_ylim(-0.5, 2.7)
    for i, src in enumerate(["f5tts", "xtts", "genuine"]):
        v = np.array([c["naturalness"]["distance"] for c in D["clips"] if c["source"] == src])
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=16, color=COL[src], edgecolor="white", linewidth=0.6, zorder=2)
        ax.plot([v.mean()] * 2, [i - 0.3, i + 0.3], color=INK, lw=1.4, zorder=3)
    ax.set_yticks([0, 1, 2], [NAME[s] for s in ["f5tts", "xtts", "genuine"]], color=INK2)
    ax.set_xlabel("naturalness distance to other speakers' readings of the same sentence (bar = mean)")
    ax.grid(axis="x", color=RULE, lw=0.4)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig-naturalness.pdf"); plt.close(fig)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    D = lab_data()
    fig_contours(D); fig_roc(D); fig_tells(); fig_naturalness(D)
    print("figures ->", OUT, sorted(os.listdir(OUT)))
