"""Strict re-evaluation of all datasets with confidence intervals and shortcut checks.

Uses features already computed per rep/recording (no labels involved in computing them):
- out/v2_features_per_rep.csv  (REHAB24-6, evaluate2.py)
- UI-PRMD features rebuilt with eval_uiprmd.py functions
- out/kimore_features.csv      (eval_kimore.py)
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd

import eval_uiprmd as U
from strict import auc, cluster_ci, lopo_predict, rho
from paths import DATA, OUT  # noqa: E402

warnings.filterwarnings("ignore")
REHAB_FEATS = ["rom", "arm_forward", "shrug", "head_tilt", "elbow_max", "elbow_min", "trunk_lean", "asym", "smooth", "duration"]


def fmt(v, ci):
    return "%.2f [%.2f-%.2f]" % (v, ci[0], ci[1])


def run_clf(df, feats, ylab, group):
    pred, chosen = lopo_predict(df[feats].to_numpy(float), df[ylab].to_numpy(), df[group].to_numpy(), "clf")
    y, g = df[ylab].to_numpy(), df[group].to_numpy()
    return auc(y, pred), cluster_ci(auc, y, pred, g), ((pred >= 0.5) == y).mean(), pred


def rehab():
    df = pd.read_csv(os.path.join(OUT, "v2_features_per_rep.csv"))
    variants = {
        "all checks": REHAB_FEATS,
        "without duration": [f for f in REHAB_FEATS if f != "duration"],
        "without elbow checks": [f for f in REHAB_FEATS if not f.startswith("elbow")],
        "without duration and elbow": [f for f in REHAB_FEATS if f != "duration" and not f.startswith("elbow")],
    }
    rows = []
    for ex, g in df.groupby("ex"):
        for vname, feats in variants.items():
            feats = [f for f in feats if not (ex == 1 and f == "asym")]
            for src in ("cam", "mocap"):
                a, ci, acc, _ = run_clf(g.assign(**{src + "_" + f: g[src + "_" + f] for f in feats}),
                                        [src + "_" + f for f in feats], "correct", "person")
                rows.append(dict(dataset="REHAB24-6", task="Ex%d" % ex, variant=vname, source=src,
                                 n_reps=len(g), n_people=g.person.nunique(), AUC=fmt(a, ci), accuracy=round(acc, 2)))
    # shortcut checks: duration alone, rep position alone, and does adding our checks
    # beat rep position (correct/incorrect reps come in blocks within each video)
    main = [f for f in REHAB_FEATS if f != "duration"]
    for ex, g in df.groupby("ex"):
        feats = [f for f in main if not (ex == 1 and f == "asym")]
        for vname, cols, src in (("duration ONLY", ["cam_duration"], "timing"),
                                 ("rep position ONLY", ["rep_position"], "order"),
                                 ("rep position + checks", ["rep_position"] + ["cam_" + f for f in feats], "cam")):
            a, ci, acc, pred = run_clf(g, cols, "correct", "person")
            rows.append(dict(dataset="REHAB24-6", task="Ex%d" % ex, variant=vname, source=src,
                             n_reps=len(g), n_people=g.person.nunique(), AUC=fmt(a, ci), accuracy=round(acc, 2)))
        # main camera score: AUC separately in early and late halves of each video
        _, _, _, pred = run_clf(g, ["cam_" + f for f in feats], "correct", "person")
        for half, m in (("first half of video", g.rep_position <= 0.5), ("second half of video", g.rep_position > 0.5)):
            y = g.correct[m].to_numpy()
            rows.append(dict(dataset="REHAB24-6", task="Ex%d" % ex, variant="main score, " + half, source="cam",
                             n_reps=int(m.sum()), n_people=g.person[m].nunique(),
                             AUC=fmt(auc(y, pred[m.to_numpy()]), cluster_ci(auc, y, pred[m.to_numpy()], g.person[m].to_numpy())),
                             accuracy=round(((pred[m.to_numpy()] >= 0.5) == y).mean(), 2)))
    c = pd.read_csv(os.path.join(OUT, "v2_results_counts.csv"))
    for (ex, src), g in c.groupby(["ex", "src"]):
        for col, name in (("n_found", "rep count, labelled span (upper bound)"), ("n_found_full_video", "rep count, whole video")):
            if col in g:
                err = (g[col] - g.n_true).abs()
                rows.append(dict(dataset="REHAB24-6", task="Ex%d" % ex, variant=name, source=src, n_reps=int(g.n_true.sum()),
                                 n_people=len(g), AUC="exact %d%%, within 1: %d%%" % (100 * (err == 0).mean(), 100 * (err <= 1).mean())))
    return rows


def uiprmd():
    recs = []
    for folder, correct in (("Segmented Movements", 1), ("Incorrect Segmented Movements", 0)):
        for p in glob.glob(os.path.join(U.BASE, folder, "Vicon", "Positions", "*.txt")):
            mv, subj, ep = map(int, re.search(r"m(\d+)_s(\d+)_e(\d+)", os.path.basename(p)).groups())
            if mv in U.MOVES:
                arm = "left" if subj in (7, 10) else "right"
                recs.append(dict(move=mv, subject=subj, correct=correct, **U.features(U.load(p), arm)))
    df = pd.DataFrame(recs)
    feats = [c for c in df.columns if c not in ("move", "subject", "correct")]
    rows = []
    for mv, g in df.groupby("move"):
        for vname, fs in (("all checks", feats), ("without duration", [f for f in feats if f != "duration"])):
            a, ci, acc, _ = run_clf(g, fs, "correct", "subject")
            rows.append(dict(dataset="UI-PRMD", task="m%02d %s" % (mv, U.MOVES[mv]), variant=vname, source="Vicon",
                             n_reps=len(g), n_people=g.subject.nunique(), AUC=fmt(a, ci), accuracy=round(acc, 2)))
        a, ci, acc, _ = run_clf(g, ["duration"], "correct", "subject")
        rows.append(dict(dataset="UI-PRMD", task="m%02d %s" % (mv, U.MOVES[mv]), variant="duration ONLY", source="Vicon",
                         n_reps=len(g), n_people=g.subject.nunique(), AUC=fmt(a, ci), accuracy=round(acc, 2)))
    return rows


def kimore():
    df = pd.read_csv(os.path.join(OUT, "kimore_features.csv"))
    feats = [c for c in df.columns if c not in ("subject", "group", "ex", "TS")]
    rows = []
    for ex, g in df.groupby("ex"):
        g = g.reset_index(drop=True)
        for vname, fs in (("all checks", feats), ("without duration and rep count", [f for f in feats if f not in ("duration", "reps")])):
            pred, _ = lopo_predict(g[fs].to_numpy(float), g.TS.to_numpy(), g.subject.to_numpy(), "reg")
            mae, base = np.abs(pred - g.TS).mean(), np.abs(g.TS - g.TS.mean()).mean()
            r, ci = rho(g.TS, pred), cluster_ci(rho, g.TS.to_numpy(), pred, g.subject.to_numpy())
            patients = ~g.group.isin(["Expert", "NotExpert"]).to_numpy()
            r_p = rho(g.TS[patients], pred[patients])
            ci_p = cluster_ci(rho, g.TS.to_numpy()[patients], pred[patients], g.subject.to_numpy()[patients])
            r_h = rho(g.TS[~patients], pred[~patients])
            rows.append(dict(dataset="KIMORE", task="Es%d" % ex, variant=vname, source="Kinect",
                             n_reps=len(g), n_people=g.subject.nunique(), rho_all=fmt(r, ci),
                             MAE_vs_mean_baseline="%.1f vs %.1f" % (mae, base),
                             rho_patients_only=fmt(r_p, ci_p), rho_healthy_only=round(r_h, 2),
                             n_patients=int(patients.sum())))
    return rows


def main():
    res = pd.DataFrame(rehab() + uiprmd() + kimore())
    pd.set_option("display.width", 250)
    for ds, g in res.groupby("dataset", sort=False):
        print("\n==", ds, "==")
        print(g.dropna(axis=1, how="all").drop(columns="dataset").to_string(index=False))
    res.to_csv(os.path.join(OUT, "strict_results.csv"), index=False)


if __name__ == "__main__":
    main()
