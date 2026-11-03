"""Paper 0, REHAB24-6 arm exercises, version 2.

Changes vs evaluate.py:
- rep counting with hysteresis (features.count_reps);
- mistake-specific features (arm forward, shoulder shrug, head tilt, range, bent elbow,
  trunk lean, left/right asymmetry, smoothness);
- the score is a logistic model trained leave-one-person-out: each person is scored by a
  model that never saw them. Score = 100 x probability "correct".
"""
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import rehab24 as R
from features import count_reps, lean_within, signals, sparc

from one_euro import OneEuro
from paths import DATA, OUT  # noqa: E402

MP = os.path.join(OUT, "mediapipe")
FPS = 30.0
SIDES = {1: ["right"], 2: ["left", "right"]}
MPI = dict(ls=11, rs=12, le=13, re=14, lw=15, rw=16, lh=23, rh=24, lear=7, rear=8)


def euro(x):
    f, y = OneEuro(), np.empty_like(x)
    for i, v in enumerate(x):
        y[i] = f(v, i / FPS) if np.isfinite(v) else (f.x if f.x is not None else np.nan)
    return y


def video_signals(ex, vid):
    a = R.mocap3d(ex, vid)
    pm = R.mocap_points(a)
    sm = signals(pm, a[:, R.J["head"]], a[:, R.J["neck"]])
    p = os.path.join(MP, "Ex%d_%s_c17.npz" % (ex, vid))
    w = np.load(p)["world"].astype(float)
    pc = {k: w[:, i] for k, i in MPI.items() if k not in ("lear", "rear")}
    sc = signals(pc, (w[:, MPI["lear"]] + w[:, MPI["rear"]]) / 2, (pc["ls"] + pc["rs"]) / 2)
    # One Euro filter on angle signals only (its settings are in degrees);
    # shoulder height (torso lengths) and torso vectors stay unfiltered
    for side in ("left", "right"):
        for k in ("elevation", "plane_dev", "elbow"):
            sc[side][k] = euro(sc[side][k])
    sc["head_tilt"] = euro(sc["head_tilt"])
    return {"mocap": sm, "cam": sc}


def rep_features(s, ex, sl):
    f = {}
    arms = SIDES[ex]
    el = np.array([s[a]["elevation"][sl] for a in arms])
    f["rom"] = np.nanmean(np.nanmax(el, axis=1))
    raised = np.nanmean(el, axis=0) > 30
    pd_ = np.array([s[a]["plane_dev"][sl] for a in arms])
    f["arm_forward"] = np.nanmean([np.nanmax(np.abs(x[raised])) if raised.any() else np.nan for x in pd_])
    f["shrug"] = np.nanmean([np.nanmax(s[a]["shoulder_h"][sl]) - np.nanmedian(s[a]["shoulder_h"][sl][:5]) for a in arms])
    ht = s["head_tilt"][sl]
    f["head_tilt"] = np.nanmax(np.abs(ht - np.nanmedian(ht[:5])))
    eb = np.array([s[a]["elbow"][sl] for a in arms])
    f["elbow_max"] = np.nanmean(np.nanmax(eb, axis=1))
    f["elbow_min"] = np.nanmean(np.nanmin(eb, axis=1))
    f["trunk_lean"] = lean_within(s["torso_up"], sl)
    f["asym"] = np.nanmean(np.abs(el[0] - el[1])) if len(arms) == 2 else 0.0
    f["smooth"] = np.nanmean([sparc(x) for x in el])
    f["duration"] = (sl.stop - sl.start) / FPS
    return f


FEATS = ["rom", "arm_forward", "shrug", "head_tilt", "elbow_max", "elbow_min", "trunk_lean", "asym", "smooth", "duration"]


def main():
    seg = R.segmentation()
    seg = seg[seg.exercise_id.isin([1, 2]) & (seg.mocap_erroneous == 0)]
    rows, counts = [], []
    for (ex, vid), reps in seg.groupby(["exercise_id", "video_id"]):
        S = video_signals(ex, vid)
        span = slice(reps.first_frame.min(), reps.last_frame.max() + 1)
        for src in ("mocap", "cam"):
            el_all = np.nanmean([S[src][a]["elevation"] for a in SIDES[ex]], axis=0)
            n_span, _ = count_reps(el_all[span])      # upper bound: exercise span known from labels
            n_full, _ = count_reps(el_all)            # whole video
            counts.append(dict(ex=ex, vid=vid, src=src, n_true=len(reps), n_found=n_span, n_found_full_video=n_full))
        order = reps.repetition_number.rank().to_numpy()
        for (_, r), pos in zip(reps.iterrows(), order):
            sl = slice(r.first_frame, r.last_frame + 1)
            row = dict(ex=ex, vid=vid, person=r.person_id, correct=r.correctness, orientation=r.cam17_orientation,
                       rep_number=r.repetition_number, rep_position=(pos - 1) / max(len(reps) - 1, 1))
            for src in ("mocap", "cam"):
                for k, v in rep_features(S[src], ex, sl).items():
                    row[src + "_" + k] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    counts = pd.DataFrame(counts)
    os.makedirs(OUT, exist_ok=True)
    df.to_csv(os.path.join(OUT, "v2_features_per_rep.csv"), index=False)

    print("== Repetition counting (hysteresis) ==")
    for src, g in counts.groupby("src"):
        print("  %-5s videos %d | exact %.0f%% | within 1 rep %.0f%% | mean |error| %.2f | total reps %d found %d"
              % (src, len(g), 100 * (g.n_found == g.n_true).mean(), 100 * ((g.n_found - g.n_true).abs() <= 1).mean(),
                 (g.n_found - g.n_true).abs().mean(), g.n_true.sum(), g.n_found.sum()))

    print("\n== Each mistake check alone: AUC (0.5 = chance; <0.5 means 'higher = more incorrect') ==")
    uni = []
    for ex, g in df.groupby("ex"):
        for k in FEATS:
            line = dict(ex=ex, feature=k)
            for src in ("mocap", "cam"):
                x = g[src + "_" + k]
                ok = np.isfinite(x)
                line[src] = roc_auc_score(g.correct[ok], x[ok]) if ok.sum() > 10 and g.correct[ok].nunique() == 2 else np.nan
            uni.append(line)
    uni = pd.DataFrame(uni)
    print(uni.round(2).pivot(index="feature", columns="ex", values=["cam", "mocap"]).to_string())

    print("\n== Combined score, leave-one-person-out (each person scored by a model that never saw them) ==")
    res, scored = [], []
    for ex, g in df.groupby("ex"):
        feats = [k for k in FEATS if not (ex == 1 and k == "asym")]
        for src in ("mocap", "cam"):
            X = g[[src + "_" + k for k in feats]].to_numpy(float)
            X = np.where(np.isfinite(X), X, np.nanmedian(X, axis=0))
            y = g.correct.to_numpy()
            prob = np.full(len(g), np.nan)
            for p in g.person.unique():
                tr, te = (g.person != p).to_numpy(), (g.person == p).to_numpy()
                model = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=1000))
                model.fit(X[tr], y[tr])
                prob[te] = model.predict_proba(X[te])[:, 1]
            auc = roc_auc_score(y, prob)
            acc = ((prob >= 0.5) == y).mean()
            res.append(dict(ex=ex, src=src, reps=len(g), AUC=auc, accuracy=acc,
                            score_correct=100 * prob[y == 1].mean(), score_incorrect=100 * prob[y == 0].mean()))
            for o in ("front", "half-profile"):
                m = (g.orientation == o).to_numpy()
                if m.sum() > 10 and len(set(y[m])) == 2:
                    res[-1]["AUC_" + o] = roc_auc_score(y[m], prob[m])
            scored.append(pd.DataFrame(dict(ex=ex, src=src, person=g.person.values, correct=y, score=100 * prob)))
    res = pd.DataFrame(res)
    print(res.round(2).to_string(index=False))
    res.to_csv(os.path.join(OUT, "v2_results_score.csv"), index=False)
    pd.concat(scored).to_csv(os.path.join(OUT, "v2_scores_per_rep.csv"), index=False)
    counts.to_csv(os.path.join(OUT, "v2_results_counts.csv"), index=False)
    uni.to_csv(os.path.join(OUT, "v2_results_single_checks.csv"), index=False)


if __name__ == "__main__":
    main()
