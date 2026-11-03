"""UI-PRMD shoulder exercises (m07 abduction, m08 extension, m09 internal-external rotation,
m10 scaption): same mistake checks + leave-one-person-out score as REHAB24-6.

Uses the Vicon marker positions (absolute, mm). The Kinect positions in UI-PRMD are
relative to the parent joint in its local frame, so they are not used.
Exercising arm per subject from the paper's Table 4: s07 and s10 left, others right.
"""
import glob
import os
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import signals, sparc
from paths import DATA, OUT  # noqa: E402

BASE = os.path.join(DATA, "UI-PRMD", "extracted")
MARK = ["LFHD", "RFHD", "LBHD", "RBHD", "C7", "T10", "CLAV", "STRN", "RBAK", "LSHO", "LUPA", "LELB", "LFRM",
        "LWRA", "LWRB", "LFIN", "RSHO", "RUPA", "RELB", "RFRM", "RWRA", "RWRB", "RFIN", "LASI", "RASI", "LPSI",
        "RPSI", "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE", "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE"]
I = {m: i for i, m in enumerate(MARK)}
MOVES = {7: "shoulder abduction", 8: "shoulder extension", 9: "shoulder int/ext rotation", 10: "shoulder scaption"}
FPS = 100.0


def load(path):
    a = np.loadtxt(path, delimiter=",").reshape(-1, 39, 3)
    g = lambda *ms: np.mean([a[:, I[m]] for m in ms], axis=0)
    p = dict(ls=g("LSHO"), rs=g("RSHO"), le=g("LELB"), re=g("RELB"), lw=g("LWRA", "LWRB"), rw=g("RWRA", "RWRB"),
             lh=g("LASI", "LPSI"), rh=g("RASI", "RPSI"))
    return signals(p, g("LFHD", "RFHD", "LBHD", "RBHD"), g("C7"))


def features(s, arm):
    f = {}
    el = s[arm]["elevation"]
    f["rom"] = np.nanmax(el)
    raised = el > 30
    f["arm_forward"] = np.nanmax(np.abs(s[arm]["plane_dev"][raised])) if raised.any() else np.nan
    f["shrug"] = np.nanmax(s[arm]["shoulder_h"]) - np.nanmedian(s[arm]["shoulder_h"][:5])
    f["head_tilt"] = np.nanmax(np.abs(s["head_tilt"] - np.nanmedian(s["head_tilt"][:5])))
    f["elbow_max"] = np.nanmax(s[arm]["elbow"])
    f["elbow_min"] = np.nanmin(s[arm]["elbow"])
    f["trunk_lean"] = np.nanmax(s["trunk_lean"])
    other = "left" if arm == "right" else "right"
    f["other_arm_move"] = np.ptp(s[other]["elevation"])
    f["smooth"] = sparc(el, fs=FPS)
    f["duration"] = len(el) / FPS
    return f


def main():
    rows = []
    for folder, correct in (("Segmented Movements", 1), ("Incorrect Segmented Movements", 0)):
        for p in glob.glob(os.path.join(BASE, folder, "Vicon", "Positions", "*.txt")):
            m = re.search(r"m(\d+)_s(\d+)_e(\d+)", os.path.basename(p))
            mv, subj, ep = map(int, m.groups())
            if mv not in MOVES:
                continue
            arm = "left" if subj in (7, 10) else "right"
            rows.append(dict(move=mv, subject=subj, episode=ep, correct=correct, **features(load(p), arm)))
    df = pd.DataFrame(rows)
    feats = [c for c in df.columns if c not in ("move", "subject", "episode", "correct")]
    out = []
    for mv, g in df.groupby("move"):
        X = g[feats].to_numpy(float)
        X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
        y = g.correct.to_numpy()
        prob = np.full(len(g), np.nan)
        for s in g.subject.unique():
            tr, te = (g.subject != s).to_numpy(), (g.subject == s).to_numpy()
            model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000)).fit(X[tr], y[tr])
            prob[te] = model.predict_proba(X[te])[:, 1]
        single = {k: roc_auc_score(y, g[k].fillna(g[k].median())) for k in feats}
        best = max(single, key=lambda k: abs(single[k] - 0.5))
        out.append(dict(move="m%02d %s" % (mv, MOVES[mv]), reps=len(g), AUC=roc_auc_score(y, prob),
                        accuracy=((prob >= 0.5) == y).mean(), score_correct=100 * prob[y == 1].mean(),
                        score_incorrect=100 * prob[y == 0].mean(), best_single_check="%s (%.2f)" % (best, single[best])))
    res = pd.DataFrame(out)
    print("== UI-PRMD (Vicon markers), leave-one-person-out score ==")
    print(res.round(2).to_string(index=False))
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "uiprmd_results.csv"), index=False)


if __name__ == "__main__":
    main()
