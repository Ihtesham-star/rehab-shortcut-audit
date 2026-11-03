"""Shortcut audit across rehabilitation datasets (paper 0).

Question per dataset: can a NON-movement variable (rep order, recording session,
patient group, seating position, duration) predict "correct vs incorrect" as well as
movement does, even under leave-one-person-out evaluation?

All models: strict.lopo_predict (imputer/scaler/C tuned inside training people only).
"""
import glob
import os
import sys
import warnings

import numpy as np
import pandas as pd

from strict import auc, cluster_ci, lopo_predict
from paths import DATA, OUT  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)


def fmt(v, ci):
    return "%.2f [%.2f-%.2f]" % (v, ci[0], ci[1])


def score(name, X, y, g, rows, dataset, task):
    """LOPO AUC with person-cluster CI; X=None means y is scored directly by g-level column."""
    pred, _ = lopo_predict(np.asarray(X, float), y, g, "clf")
    rows.append(dict(dataset=dataset, task=task, model=name, n=len(y), people=len(set(g)),
                     AUC=fmt(auc(y, pred), cluster_ci(auc, y, pred, g))))
    return pred


def generic_motion(a):
    """Movement features from a (frames, joints, 3) skeleton: centred on the root joint,
    scaled by torso length; per joint/axis range (p95-p5) and mean speed."""
    a = a - a[:, :1]
    torso = np.nanmedian(np.linalg.norm(a[:, 1], axis=1)) or 1.0
    a = a / torso
    rng = np.nanpercentile(a, 95, axis=0) - np.nanpercentile(a, 5, axis=0)
    spd = np.nanmean(np.abs(np.diff(a, axis=0)), axis=0)
    return np.concatenate([rng.ravel(), spd.ravel()])


# ---------------------------------------------------------------- IntelliRehabDS
def irds(rows):
    base = os.path.join(DATA, "IntelliRehabDS", "extracted", "SkeletonData", "Simplified")
    recs = []
    for p in sorted(glob.glob(os.path.join(base, "*.txt"))):
        s, d, gest, rep, lab, pos = os.path.basename(p)[:-4].split("_")
        if lab not in ("1", "2"):
            continue
        v = np.loadtxt(p, delimiter=",", ndmin=2)
        if v.shape[1] < 75 or len(v) < 16:
            continue
        a = v[:, :75].reshape(len(v), 25, 3)
        # root = SpineBase (0); joint 1 = SpineMid gives torso scale
        recs.append(dict(subject=s, gesture=int(gest), rep=int(rep), correct=int(lab == "1"),
                         position=pos.lower(), patient=int(s.startswith("2")), frames=len(v),
                         motion=generic_motion(a)))
    df = pd.DataFrame(recs)
    print("IRDS sequences", len(df), "incorrect", (df.correct == 0).sum(),
          "| incorrect from patients: %.0f%%" % (100 * df[df.correct == 0].patient.mean()))
    allpred = {k: np.full(len(df), np.nan) for k in ("motion", "position", "duration", "rep")}
    for gest, g in df.groupby("gesture"):
        idx = g.index.to_numpy()
        y, grp = g.correct.to_numpy(), g.subject.to_numpy()
        if len(set(y)) < 2:
            continue
        feats = dict(motion=np.vstack(g.motion), position=pd.get_dummies(g.position).to_numpy(float),
                     duration=g[["frames"]].to_numpy(float), rep=g[["rep"]].to_numpy(float))
        for k, X in feats.items():
            allpred[k][idx] = lopo_predict(X, y, grp, "clf")[0]
    y, grp = df.correct.to_numpy(), df.subject.to_numpy()
    ok = np.isfinite(allpred["motion"])
    for k, name in (("motion", "movement features (all people)"), ("position", "seating position ONLY"),
                    ("duration", "duration ONLY"), ("rep", "repetition number ONLY")):
        rows.append(dict(dataset="IntelliRehabDS", task="9 gestures pooled", model=name, n=int(ok.sum()),
                         people=len(set(grp[ok])), AUC=fmt(auc(y[ok], allpred[k][ok]),
                                                          cluster_ci(auc, y[ok], allpred[k][ok], grp[ok]))))
    # patient group as a score: "patient" -> predicts incorrect
    rows.append(dict(dataset="IntelliRehabDS", task="9 gestures pooled", model="patient group ONLY (no learning)",
                     n=int(ok.sum()), people=len(set(grp[ok])),
                     AUC=fmt(auc(y[ok], -df.patient.to_numpy()[ok]), cluster_ci(auc, y[ok], -df.patient.to_numpy()[ok], grp[ok]))))
    m = ok & (df.patient.to_numpy() == 1)
    rows.append(dict(dataset="IntelliRehabDS", task="9 gestures pooled", model="movement features, PATIENTS ONLY (same predictions)",
                     n=int(m.sum()), people=len(set(grp[m])),
                     AUC=fmt(auc(y[m], allpred["motion"][m]), cluster_ci(auc, y[m], allpred["motion"][m], grp[m]))))
    # retrain on patients only (no healthy people in training either)
    pp = np.full(len(df), np.nan)
    for gest, g in df[df.patient == 1].groupby("gesture"):
        if g.correct.nunique() < 2 or g.subject.nunique() < 3:
            continue
        pp[g.index.to_numpy()] = lopo_predict(np.vstack(g.motion), g.correct.to_numpy(), g.subject.to_numpy(), "clf")[0]
    m2 = np.isfinite(pp)
    rows.append(dict(dataset="IntelliRehabDS", task="9 gestures pooled", model="movement features, trained+tested on PATIENTS ONLY",
                     n=int(m2.sum()), people=len(set(grp[m2])),
                     AUC=fmt(auc(y[m2], pp[m2]), cluster_ci(auc, y[m2], pp[m2], grp[m2]))))
    return df


# ---------------------------------------------------------------- UI-PRMD
STATIC_PAIRS = [("LASI", "RASI"), ("LPSI", "RPSI"), ("LKNE", "RKNE"), ("LANK", "RANK"), ("LTHI", "LKNE"),
                ("RTHI", "RKNE"), ("LTIB", "LANK"), ("RTIB", "RANK"), ("LFHD", "RFHD"), ("LBHD", "RBHD"), ("C7", "T10")]


def uiprmd_static(path):
    """Non-movement features from the first 10 frames (shoulder exercises: legs, pelvis and
    head are still). Marker spacing on rigid segments changes only if markers were re-applied
    (a recording-session fingerprint); pelvis location/orientation = where the person stood."""
    import eval_uiprmd as U
    a = np.loadtxt(path, delimiter=",").reshape(-1, 39, 3)[:10].mean(axis=0)
    m = lambda k: a[U.I[k]]
    f = [np.linalg.norm(m(p) - m(q)) for p, q in STATIC_PAIRS]
    pelvis = np.mean([m(k) for k in ("LASI", "RASI", "LPSI", "RPSI")], axis=0)
    d = m("RASI") - m("LASI")
    return np.concatenate([f, pelvis, d / np.linalg.norm(d)])


RIGID_PAIRS = [("LASI", "RASI"), ("LPSI", "RPSI"), ("LASI", "LPSI"), ("RASI", "RPSI"), ("LFHD", "RFHD"),
               ("LBHD", "RBHD"), ("LFHD", "LBHD"), ("RFHD", "RBHD"), ("LTHI", "LKNE"), ("RTHI", "RKNE"),
               ("LTIB", "LANK"), ("RTIB", "RANK"), ("LHEE", "LTOE"), ("RHEE", "RTOE")]


def uiprmd_rigid(path):
    """Strictest version: only marker spacings WITHIN rigid segments (pelvis, head, thigh, shank,
    foot), first 10 frames. These cannot change with how the exercise is performed; they change
    only when markers are re-applied, i.e. they fingerprint the recording session."""
    import eval_uiprmd as U
    a = np.loadtxt(path, delimiter=",").reshape(-1, 39, 3)[:10].mean(axis=0)
    return np.array([np.linalg.norm(a[U.I[p]] - a[U.I[q]]) for p, q in RIGID_PAIRS])


def within_person_auc(X, y, g):
    """Leave-one-episode-out inside each person (no other people involved); mean per-person AUC."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    aucs = []
    for p in np.unique(g):
        m = g == p
        Xp, yp = X[m], y[m]
        if len(set(yp)) < 2:
            continue
        pr = np.empty(len(yp))
        for i in range(len(yp)):
            tr = np.arange(len(yp)) != i
            pr[i] = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xp[tr], yp[tr]).predict_proba(Xp[i:i + 1])[0, 1]
        aucs.append(auc(yp, pr))
    return np.mean(aucs), np.percentile(aucs, [0, 100])


def uiprmd(rows):
    import re
    import eval_uiprmd as U
    recs = []
    for folder, correct in (("Segmented Movements", 1), ("Incorrect Segmented Movements", 0)):
        for p in glob.glob(os.path.join(U.BASE, folder, "Vicon", "Positions", "*.txt")):
            mv, subj, ep = map(int, re.search(r"m(\d+)_s(\d+)_e(\d+)", os.path.basename(p)).groups())
            if mv in U.MOVES:
                arm = "left" if subj in (7, 10) else "right"
                f = U.features(U.load(p), arm)
                f.pop("duration")
                recs.append(dict(move=mv, subject=subj, correct=correct, static=uiprmd_static(p), rigid=uiprmd_rigid(p),
                                 motion=np.array(list(f.values()), float)))
    df = pd.DataFrame(recs)
    for mv, g in df.groupby("move"):
        y, grp = g.correct.to_numpy(), g.subject.to_numpy()
        task = "m%02d %s" % (mv, U.MOVES[mv])
        for name, col in (("our movement checks", "motion"), ("STATIC non-movement (marker spacing, stance)", "static"),
                          ("RIGID-segment marker spacing only", "rigid")):
            X = np.vstack(g[col])
            score(name + ", LOPO", X, y, grp, rows, "UI-PRMD", task)
            m, (lo, hi) = within_person_auc(X, y, grp)
            rows.append(dict(dataset="UI-PRMD", task=task, model=name + ", within each person",
                             n=len(y), people=len(set(grp)), AUC="%.2f (people range %.2f-%.2f)" % (m, lo, hi)))
    return df


# ---------------------------------------------------------------- KIMORE
def kimore(rows):
    df = pd.read_csv(os.path.join(OUT, "kimore_features.csv"))
    from strict import rho
    feats = [c for c in df.columns if c not in ("subject", "group", "ex", "TS", "duration", "reps")]
    for ex, g in df.groupby("ex"):
        g = g.reset_index(drop=True)
        patient = (~g.group.isin(["Expert", "NotExpert"])).astype(float).to_numpy()
        y, grp = g.TS.to_numpy(), g.subject.to_numpy()
        pred, _ = lopo_predict(g[feats].to_numpy(float), y, grp, "reg")
        for name, p in (("our movement features (ridge, LOPO)", pred), ("patient-vs-healthy label ONLY", -patient)):
            rows.append(dict(dataset="KIMORE", task="Es%d" % ex, model=name, n=len(y), people=len(set(grp)),
                             AUC="rho " + fmt(rho(y, p), cluster_ci(rho, y, p, grp))))
        m = patient == 1
        rows.append(dict(dataset="KIMORE", task="Es%d" % ex, model="our movement features, patients only",
                         n=int(m.sum()), people=int(m.sum()),
                         AUC="rho " + fmt(rho(y[m], pred[m]), cluster_ci(rho, y[m], pred[m], grp[m]))))


def main(which):
    rows = []
    if "irds" in which:
        irds(rows)
    if "uiprmd" in which:
        uiprmd(rows)
    if "kimore" in which:
        kimore(rows)
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_colwidth", 70)
    print(res.to_string(index=False))
    os.makedirs(OUT, exist_ok=True)
    res.to_csv(os.path.join(OUT, "shortcuts_%s.csv" % "_".join(which)), index=False)


if __name__ == "__main__":
    main(sys.argv[1:] or ["irds"])
