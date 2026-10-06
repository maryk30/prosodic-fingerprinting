"""Export the data behind VoiceGuard Lab (web/lab.html).

Runs every held-out clip (s21–s30: each speaker's genuine reading plus its
XTTS-v2 and F5-TTS clones) through the real pipeline and writes what each
stage produced: VAD speech/pause segments, pitch and loudness contours, the
feature values and their z-scores against the claimed speaker's fingerprint,
component distances, the blended verdict, and the naturalness check.

Only derived numbers leave this script — no audio — so the page can be shared
without handing out voice samples someone could clone from.

    python src/export_viz.py        # -> web/lab_data.js (window.VG_DATA = {...})

Needs enrolled models (cli.py enroll ... s01–s20) and naturalness thresholds
(python src/contour.py).
"""

import json
import os
import sys

import librosa
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "models"))
from clone_common import CLONE_SENTENCES, sentences  # noqa: E402
from contour import ContourCache, clone_takes, enrolled_speakers, find_take, load_thresholds, naturalness  # noqa: E402
from distance import DEFAULT_COMPONENTS, FingerprintDistanceDetector  # noqa: E402
from eval import evaluate  # noqa: E402
from features import _internal_pauses, _pitch_track, extract_raw_features, speech_mask, to_row  # noqa: E402
from fingerprint import PROSODIC_FEATURES, REGISTER_FEATURES, VOICE_QUALITY_FEATURES  # noqa: E402
from impostor_eval import SYSTEMS, clone_trials, load_table, summarize, trial_scores  # noqa: E402
from preprocessing import preprocess  # noqa: E402
from spectral_features import extract_mfcc_row  # noqa: E402

OUT = "web/lab_data.js"
CONTOUR_HOP_S = 0.032  # export every 2nd pYIN frame — plenty for a chart, half the size

LABELS = {
    "f0_std": "Pitch variability", "f0_range": "Pitch range", "f0_slope": "Pitch slope",
    "f0_velocity": "Pitch velocity", "f0_final_move": "Final rise/fall",
    "energy_std": "Loudness variability", "energy_slope": "Loudness slope",
    "speaking_rate_mean": "Speaking rate", "npvi": "Rhythm (nPVI)", "voiced_fraction": "Voiced fraction",
    "varco_v": "Vowel-length variation", "varco_uv": "Consonant-length variation",
    "pause_count": "Pause count", "pause_mean_dur": "Pause length", "pause_var_dur": "Pause variability",
    "jitter": "Jitter", "shimmer": "Shimmer", "f0_mean": "Pitch level",
}
GROUP = {f: "prosody" for f in PROSODIC_FEATURES} | {f: "voice_quality" for f in VOICE_QUALITY_FEATURES} \
    | {f: "pitch_register" for f in REGISTER_FEATURES}


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) else round(float(x), nd)


def contours(path: str, f0_ref_hz: float) -> dict:
    clip = preprocess(path)
    times, f0, voiced = _pitch_track(clip)
    voiced = voiced & speech_mask(times, clip)  # same speech-only rule as the features
    step = max(1, int(round(CONTOUR_HOP_S / (times[1] - times[0]))))
    st = np.where(voiced, 12 * np.log2(np.where(voiced, f0, 1.0) / f0_ref_hz), np.nan)
    hop = int(round((times[1] - times[0]) * clip.sr))
    rms = librosa.feature.rms(y=clip.audio, frame_length=2 * hop, hop_length=hop)[0][: len(times)]
    db = 20 * np.log10(rms + 1e-9)
    db = db - np.median(db[voiced]) if voiced.any() else db - np.median(db)
    return {
        "duration": r(len(clip.audio) / clip.sr, 2),
        "t": [r(t, 3) for t in times[::step]],
        "pitch": [r(v, 2) for v in st[::step]],  # semitones vs the claimed speaker's median pitch
        "loud": [r(v, 1) for v in db[::step]],  # dB vs the clip's median voiced loudness
        "speech": [[r(s.start, 2), r(s.end, 2)] for s in clip.speech_segments],
        "pauses": len(_internal_pauses(clip)),
    }


def clip_record(path, speaker, source, n, det, thr, cache, speakers) -> dict:
    raw = extract_raw_features(path, speaker, "candidate", source)
    row = to_row(raw, det.f0_ref_hz)
    row.update(extract_mfcc_row(path, speaker, "candidate", source))
    res = det.score(row)

    feats = []
    for name, (cols, _, _) in det.components.items():
        if name == "timbre":
            continue  # 26 MFCC stats: summarized by the component distance only
        mean, std = det.stats[name]
        for c, m, s in zip(cols, mean, std):
            v = row[c]
            feats.append({"key": c, "label": LABELS[c], "group": GROUP[c], "value": r(v, 4),
                          "mean": r(m, 4), "sd": r(s, 4),
                          "z": r((v - m) / s, 2) if v is not None and np.isfinite(v) else None})

    nat = naturalness(cache(path), n, speaker, speakers, cache)
    return {
        "id": f"{speaker}-{source}-s{n:02d}", "speaker": speaker, "source": source, "sentence": n,
        **contours(path, det.f0_ref_hz),
        "features": feats,
        "components": [
            {"name": name, "weight": w, "cap": cap, "distance": r(res["components"][name]["distance"]),
             "capped": res["components"][name]["capped"], "share": r(res["components"][name]["share"])}
            for name, (_, w, cap) in det.components.items()
        ],
        "distance": r(res["distance"]), "threshold": r(res["threshold"]), "verdict": res["prediction"],
        "naturalness": {"distance": r(nat, 6), "threshold": r(thr[speaker], 6),  # full precision: ranks feed ROC-AUC
                        "verdict": "genuine" if nat <= thr[speaker] else "synthetic"},
    }


def summary_tables() -> dict:
    df, f0_refs = load_table(include_synthetic=True)
    out = {}
    for name in ["blend (default, capped)", "prosody only", "register + timbre"]:
        t = clone_trials(df, f0_refs, SYSTEMS[name])
        g = t[t["is_genuine"] == 1]
        out[name] = {
            tts: {k: r(v) for k, v in summarize(pd.concat([g, t[t["source"] == tts]])).items()
                  if k in ("roc_auc", "eer", "genuine_accept", "impostor_reject")}
            for tts in ["xtts", "f5tts"]
        }
    gen = df[df["label"] == "genuine"]
    imp = summarize(trial_scores(gen, f0_refs, SYSTEMS["blend (default, capped)"]))
    out["impostor_blend"] = {k: r(v) for k, v in imp.items() if k in ("roc_auc", "eer")}
    return out


def main() -> None:
    speakers = enrolled_speakers()
    text = sentences()
    thr = load_thresholds()
    cache = ContourCache()
    clips = []
    for spk in speakers:
        det = FingerprintDistanceDetector.load(spk)
        for n in CLONE_SENTENCES:
            cands = [("genuine", find_take(spk, n))] + clone_takes(spk, n)
            for source, path in cands:
                if path is None or librosa.get_duration(path=path) > 15:
                    continue  # same skip rule as the pipeline (e.g. one 21 s XTTS babble)
                clips.append(clip_record(path, spk, source, n, det, thr, cache, speakers))
                print(f"{clips[-1]['id']}: {clips[-1]['verdict']} / naturalness {clips[-1]['naturalness']['verdict']}")

    nat_rows = [{"is_genuine": int(c["source"] == "genuine"), "source": c["source"],
                 "score": c["naturalness"]["threshold"] - c["naturalness"]["distance"]} for c in clips]
    nt = pd.DataFrame(nat_rows)
    nat_summary = {}
    for tts in ["xtts", "f5tts"]:
        sub = nt[nt["source"].isin(["genuine", tts])]
        m = evaluate(sub["is_genuine"].to_numpy(), sub["score"].to_numpy())
        nat_summary[tts] = {"roc_auc": r(m["roc_auc"]), "eer": r(m["eer"]),
                            "genuine_accept": r((sub.query("is_genuine == 1")["score"] >= 0).mean()),
                            "impostor_reject": r((sub.query("is_genuine == 0")["score"] < 0).mean())}

    data = {
        "speakers": speakers,
        "sentences": {n: text[n] for n in CLONE_SENTENCES},
        "weights": {name: {"weight": w, "cap": cap} for name, (_, w, cap) in DEFAULT_COMPONENTS.items()},
        "clips": clips,
        "summary": {**summary_tables(), "naturalness": nat_summary},
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write("window.VG_DATA = ")
        json.dump(data, f, separators=(",", ":"))
        f.write(";\n")
    print(f"\n{len(clips)} clips -> {OUT} ({os.path.getsize(OUT) / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
