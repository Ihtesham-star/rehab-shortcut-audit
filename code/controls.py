"""Controls for the shortcut audit (paper 0).

1. uiprmd_perm  - random-label control for the within-person rigid-marker fingerprint test:
                  shuffle labels inside each person, rerun, 200 times. Real finding -> shuffled ~0.5.
2. kimore_take  - 9 KIMORE exercise folders hold two different takes; eval_kimore.py uses the last.
                  Rebuild features with the FIRST take and compare rho (all / patients only / label-only).
3. context      - does a model that also sees the NEIGHBOURING reps (as sliding-window or whole-recording
                  models do) exploit the block order? Single-rep vs rep + previous/next rep features, LOPO.
"""
import glob
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd

from strict import auc, cluster_ci, lopo_predict, rho
from paths import DATA, OUT  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(__file__)



def fmt(v, ci):
    return "%.2f [%.2f-%.2f]" % (v, ci[0], ci[1])


# ---------------------------------------------------------------- 1
def uiprmd_perm(n_perm=200):
    import eval_uiprmd as U
    from shortcuts import uiprmd_rigid, within_person_auc
    recs = []
    for folder, correct in (("Segmented Movements", 1), ("Incorrect Segmented Movements", 0)):
        for p in glob.glob(os.path.join(U.BASE, folder, "Vicon", "Positions", "*.txt")):
            mv, subj, ep = map(int, re.search(r"m(\d+)_s(\d+)_e(\d+)", os.path.basename(p)).groups())
            if mv in U.MOVES:
                recs.append(dict(move=mv, subject=subj, correct=correct, rigid=uiprmd_rigid(p)))
    df = pd.DataFrame(recs)
    rng = np.random.default_rng(0)
    rows, nulls = [], {}
    for mv, g in df.groupby("move"):
        X, y, grp = np.vstack(g.rigid), g.correct.to_numpy(), g.subject.to_numpy()
        real, _ = within_person_auc(X, y, grp)
        null = []
        for _ in range(n_perm):
            ys = y.copy()
            for s in np.unique(grp):
                m = grp == s
                ys[m] = rng.permutation(ys[m])
            null.append(within_person_auc(X, ys, grp)[0])
        null = np.array(null)
        nulls["m%02d_null" % mv], nulls["m%02d_real" % mv] = null, np.array([real])
        rows.append(dict(task="m%02d %s" % (mv, U.MOVES[mv]), real_AUC=round(real, 3),
                         shuffled_mean=round(null.mean(), 3), shuffled_95pct=round(np.percentile(null, 95), 3),
                         shuffled_max=round(null.max(), 3), p_value="<%.3f" % (1 / n_perm) if (null >= real).sum() == 0
                         else "%.3f" % ((null >= real).mean())))
        print(rows[-1], flush=True)
    np.savez(os.path.join(OUT, "uiprmd_perm_null.npz"), **nulls)  # for Fig. 3b
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 2
def kimore_take():
    import eval_kimore as K
    rows, changed = [], []
    for subj in glob.glob(os.path.join(K.BASE, "*", "*", "*")):
        group = os.path.relpath(subj, K.BASE).split(os.sep)[1]
        ts = K.clinical(subj)
        if not ts:
            continue
        for ex in K.EX:
            jp = glob.glob(os.path.join(subj, "Es%d" % ex, "Raw", "JointPosition*.csv"))
            takes = sorted(p for p in jp if "(1)" not in os.path.basename(p))   # drop browser re-download copies
            if not jp:
                continue
            pick = takes[0] if len(takes) > 1 else jp[0]   # the main analysis (eval_kimore.py) uses the last take
            if len(takes) > 1:
                changed.append((os.path.basename(subj), ex))
            a = K.load_positions(pick)
            if np.isfinite(a).all(axis=(1, 2)).sum() < 60 or ex not in ts or not np.isfinite(ts[ex]):
                continue
            rows.append(dict(subject=os.path.basename(subj), group=group, ex=ex, TS=ts[ex], **K.features(a)))
    print("recordings switched to the other take:", changed)
    other = pd.DataFrame(rows)
    other.to_csv(os.path.join(OUT, "kimore_features_othertake.csv"), index=False)
    main = pd.read_csv(os.path.join(OUT, "kimore_features.csv"))
    res = []
    for name, df in (("last take (main)", main), ("first take (other)", other)):
        feats = [c for c in df.columns if c not in ("subject", "group", "ex", "TS", "duration", "reps")]
        for ex, g in df.groupby("ex"):
            g = g.reset_index(drop=True)
            y, grp = g.TS.to_numpy(), g.subject.to_numpy()
            pred, _ = lopo_predict(g[feats].to_numpy(float), y, grp, "reg")
            pat = (~g.group.isin(["Expert", "NotExpert"])).to_numpy()
            res.append(dict(version=name, task="Es%d" % ex, n=len(g), rho_all=fmt(rho(y, pred), cluster_ci(rho, y, pred, grp)),
                            rho_patients=round(rho(y[pat], pred[pat]), 2), rho_label_only=round(rho(y, -pat.astype(float)), 2)))
            print(res[-1], flush=True)
    return pd.DataFrame(res)


# ---------------------------------------------------------------- 3
def with_neighbours(df, feats, order_cols, sort_col):
    """Append previous- and next-rep features (within the same recording/set); edges get NaN (imputed in-fold)."""
    df = df.sort_values(order_cols + [sort_col]).copy()
    for k, sh in (("prev", 1), ("next", -1)):
        nb = df.groupby(order_cols)[feats].shift(sh)
        nb.columns = [k + "_" + c for c in feats]
        df = pd.concat([df, nb], axis=1)
    return df, feats + ["prev_" + c for c in feats] + ["next_" + c for c in feats]


def context():
    rows = []
    d = pd.read_csv(os.path.join(OUT, "v2_features_per_rep.csv"))
    F = ["rom", "arm_forward", "shrug", "head_tilt", "elbow_max", "elbow_min", "trunk_lean", "asym", "smooth"]
    for ex, g in d.groupby("ex"):
        feats = ["cam_" + f for f in F if not (ex == 1 and f == "asym")]
        g2, feats2 = with_neighbours(g, feats, ["vid"], "rep_number")
        for name, gg, fs in (("single rep", g, feats), ("rep + previous/next rep", g2, feats2)):
            y, grp = gg.correct.to_numpy(), gg.person.to_numpy()
            p, _ = lopo_predict(gg[fs].to_numpy(float), y, grp, "clf")
            rows.append(dict(dataset="REHAB24-6 (camera checks)", task="Ex%d" % ex, model=name, AUC=fmt(auc(y, p), cluster_ci(auc, y, p, grp))))
            print(rows[-1], flush=True)
    import rehab26_shortcuts as R
    s = pd.read_csv(os.path.join(R.BASE, "Segmentation.csv"), sep=";")
    s["correct"] = (s.correctness_score == 1).astype(int)
    for ex, g in s.groupby("exercise_id"):
        mot = []
        for _, r in g.iterrows():
            o = r.person_orientation
            cam = 1 + int(np.argmin([abs(-o), abs(45 - o), abs(90 - o)]))
            mot.append(R.motion_features(R.arr(r.video_id, ex, cam, "3d"), r.first_frame, r.last_frame))
        M = np.vstack(mot)
        cols = ["m%d" % i for i in range(M.shape[1])]
        g = pd.concat([g.reset_index(drop=True), pd.DataFrame(M, columns=cols)], axis=1)
        g2, cols2 = with_neighbours(g, cols, ["video_id", "person_orientation"], "first_frame")
        for name, gg, fs in (("single rep", g, cols), ("rep + previous/next rep", g2, cols2)):
            y, grp = gg.correct.to_numpy(), gg.person.to_numpy()
            p, _ = lopo_predict(gg[fs].to_numpy(float), y, grp, "clf")
            rows.append(dict(dataset="REHAB26 (generic MediaPipe)", task="Ex%d" % ex, model=name, AUC=fmt(auc(y, p), cluster_ci(auc, y, p, grp))))
            print(rows[-1], flush=True)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    for which in sys.argv[1:]:
        res = {"uiprmd_perm": uiprmd_perm, "kimore_take": kimore_take, "context": context}[which]()
        print("\n==", which, "==\n" + res.to_string(index=False))
        res.to_csv(os.path.join(OUT, "control_%s.csv" % which), index=False)
