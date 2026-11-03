"""KIMORE: does our movement-quality measure agree with clinicians' scores?

Exercises: Es1 lifting of the arms, Es2 lateral trunk tilt with arms extended (arm-related);
Es3-5 (trunk rotation, pelvis rotation, squat) are reported for completeness.
Data: Kinect v2 JointPosition*.csv (25 joints x [x, y, z, state]); clinical score
"TS" 0-50 per exercise (PO + CF) from ClinicalAssessment_*.xlsx.
Model: ridge regression on our features, leave-one-person-out; report Spearman rho
between predicted and clinician TS, and each feature's own correlation.
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import count_reps, signals, sparc
from paths import DATA, OUT  # noqa: E402

warnings.filterwarnings("ignore")
BASE = os.path.join(DATA, "KIMORE", "extracted", "KiMoRe")
K = dict(ls=4, rs=8, le=5, re=9, lw=6, rw=10, lh=12, rh=16)
EX = {1: "lifting of the arms", 2: "lateral trunk tilt, arms extended", 3: "trunk rotation",
      4: "pelvis rotation", 5: "squat"}


def load_positions(path):
    """(frames, 25, 3). Empty/broken lines become NaN frames, so time is not compressed.
    Lines with several tracked bodies: keep the body whose SpineBase is closest to the
    previously kept one (the patient), not simply the first (review 2026-10-31)."""
    frames, prev = [], None
    with open(path) as fh:
        for line in fh:
            v = [x for x in line.strip().split(",") if x != ""]
            bodies = []
            for b in range(len(v) // 100):
                try:
                    bodies.append(np.array(v[100 * b:100 * (b + 1)], float).reshape(25, 4)[:, :3])
                except ValueError:
                    pass
            bodies = [b for b in bodies if np.isfinite(b).all()]
            if not bodies:
                frames.append(np.full((25, 3), np.nan))
                continue
            pick = bodies[0] if prev is None else min(bodies, key=lambda b: np.linalg.norm(b[0] - prev))
            prev = pick[0]
            frames.append(pick)
    a = np.array(frames)
    # trim leading/trailing empty frames only
    ok = np.isfinite(a).all(axis=(1, 2))
    if ok.any():
        a = a[np.argmax(ok): len(ok) - np.argmax(ok[::-1])]
    return a


def features(a):
    p = {k: a[:, i] for k, i in K.items()}
    s = signals(p, a[:, 3], a[:, 2])   # head axis: Head - Neck
    f = {}
    el = np.array([s[side]["elevation"] for side in ("left", "right")])
    f["rom"] = np.nanmean(np.nanpercentile(el, 95, axis=1))
    f["asym"] = np.nanmean(np.abs(el[0] - el[1]))
    raised = np.nanmean(el, axis=0) > 30
    f["arm_forward"] = np.nanmean([np.nanpercentile(np.abs(s[side]["plane_dev"][raised]), 90) if raised.sum() > 5 else np.nan
                                   for side in ("left", "right")])
    f["elbow"] = np.nanmean([np.nanpercentile(s[side]["elbow"], 90) for side in ("left", "right")])
    f["shrug"] = np.nanmean([np.nanpercentile(s[side]["shoulder_h"], 95) - np.nanpercentile(s[side]["shoulder_h"], 5)
                             for side in ("left", "right")])
    f["head_tilt"] = np.nanpercentile(np.abs(s["head_tilt"] - np.nanmedian(s["head_tilt"])), 90)
    f["trunk_lean"] = np.nanpercentile(s["trunk_lean"], 90)
    f["smooth"] = np.nanmean([sparc(x) for x in el])
    f["reps"], _ = count_reps(np.nanmean(el, axis=0))
    f["duration"] = len(a) / 30.0
    return f


def clinical(subject_dir):
    xs = glob.glob(os.path.join(subject_dir, "*", "Label", "ClinicalAssessment_*.xlsx"))
    if not xs:
        return None
    x = pd.read_excel(xs[0]).iloc[0]
    return {int(re.search(r"Ex#(\d)", c).group(1)): float(x[c]) for c in x.index if "TS" in str(c)}


def main():
    rows = []
    for subj in glob.glob(os.path.join(BASE, "*", "*", "*")):
        group = os.path.relpath(subj, BASE).split(os.sep)[1]
        ts = clinical(subj)
        if not ts:
            continue
        for ex in EX:
            jp = glob.glob(os.path.join(subj, "Es%d" % ex, "Raw", "JointPosition*.csv"))
            if not jp:
                continue
            # some folders hold 2 takes (+ browser "(1)" copies): use the LAST take, i.e. the retake.
            # The first take is sometimes aborted (e.g. S_ID5 Es1: 2 s) (control check 2026-10-31).
            takes = sorted(p for p in jp if "(1)" not in os.path.basename(p)) or jp
            a = load_positions(takes[-1])
            if np.isfinite(a).all(axis=(1, 2)).sum() < 60 or ex not in ts or not np.isfinite(ts[ex]):
                continue
            rows.append(dict(subject=os.path.basename(subj), group=group, ex=ex, TS=ts[ex], **features(a)))
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c not in ("subject", "group", "ex", "TS")]
    out, single = [], []
    for ex, g in df.groupby("ex"):
        X = g[feats].to_numpy(float)
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        y = g.TS.to_numpy()
        pred = np.full(len(g), np.nan)
        for s in g.subject.unique():
            tr, te = (g.subject != s).to_numpy(), (g.subject == s).to_numpy()
            m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(X[tr], y[tr])
            pred[te] = m.predict(X[te])
        rho, pv = spearmanr(pred, y)
        mae = np.abs(pred - y).mean()
        healthy = g.group.isin(["Expert", "NotExpert"]).to_numpy()
        out.append(dict(ex="Es%d %s" % (ex, EX[ex]), people=len(g), TS_mean=y.mean(), TS_sd=y.std(),
                        rho=rho, p=pv, MAE=mae, baseline_MAE=np.abs(y - y.mean()).mean(),
                        TS_healthy=y[healthy].mean(), TS_patients=y[~healthy].mean()))
        for k in feats:
            r, p = spearmanr(g[k], y, nan_policy="omit")
            single.append(dict(ex=ex, feature=k, rho=r))
    res = pd.DataFrame(out)
    print("== KIMORE: predicted vs clinician total score (0-50), leave-one-person-out ==")
    print(res.round(3).to_string(index=False))
    sg = pd.DataFrame(single).pivot(index="feature", columns="ex", values="rho")
    print("\n== Each feature alone: Spearman rho with clinician score ==")
    print(sg.round(2).to_string())
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "kimore_results.csv"), index=False)
    df.to_csv(os.path.join(OUT, "kimore_features.csv"), index=False)


if __name__ == "__main__":
    main()
