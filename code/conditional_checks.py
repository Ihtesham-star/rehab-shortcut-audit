"""Extra analyses asked for in the pre-submission review of Paper 0 (2026-11-01).

1. IRDS: does the model score add information beyond patient status?
   - logistic regression label ~ patient + score (odds ratio of the score, person-cluster bootstrap CI);
   - cross-validated AUC of patient-only vs patient + score (grouped by person), paired bootstrap of the gain;
   - paired bootstrap CI of (deep AUC - patient-status AUC);
   - AUC per gesture (deep models and patient status).
2. IRDS: can patient status be decoded from the skeleton? Classifier trained on CORRECT repetitions only
   (generic movement features, leave-one-person-out per gesture), target = patient vs healthy.
3. KIMORE: partial Spearman rho between prediction and clinical score, controlling for patient status.
4. UI-PRMD: person-level bootstrap CI of the mean within-person rigid-spacing AUC (for Fig. 3b).
5. REHAB26: random-split AUC (from deep/run.py rehab26_random, if it has been run).

Inputs: out/deep/*.csv (out-of-fold predictions), the raw IRDS and UI-PRMD files.
Outputs: out/review_*.csv
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from strict import auc, cluster_ci, lopo_predict
from paths import DATA, OUT  # noqa: E402

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))

DEEP = os.path.join(OUT, "deep")
N_BOOT = 1000


def fmt(v, ci):
    return "%.2f [%.2f, %.2f]" % (v, ci[0], ci[1])


def boot_people(groups, n=N_BOOT, seed=0):
    """Yield index arrays of person-level bootstrap resamples."""
    rng = np.random.default_rng(seed)
    ids = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ids}
    for _ in range(n):
        yield np.concatenate([idx[g] for g in rng.choice(ids, len(ids), replace=True)])


def logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


# ------------------------------------------------------------------ 1. IRDS conditional tests
def irds_conditional():
    d = pd.read_csv(os.path.join(DEEP, "irds.csv"))
    d = d[d.protocol == "person"].reset_index(drop=True)
    rows = []
    for arch, g in d.groupby("arch"):
        g = g.reset_index(drop=True)
        y, pat, s, grp = g.y.to_numpy(), g.patient.to_numpy(), logit(g.pred.to_numpy()), g.person.astype(str).to_numpy()
        # (a) odds ratio of a 1-SD change in the logit score, adjusted for patient status
        z = (s - s.mean()) / s.std()
        X = np.column_stack([pat, z])
        coef = LogisticRegression(C=1e6, max_iter=2000).fit(X, y).coef_[0][1]
        boots = []
        for b in boot_people(grp):
            if len(np.unique(y[b])) < 2:
                continue
            boots.append(LogisticRegression(C=1e6, max_iter=2000).fit(X[b], y[b]).coef_[0][1])
        or_ci = np.exp(np.percentile(boots, [2.5, 97.5]))
        # (b) cross-validated AUC: patient only vs patient + score (grouped by person)
        p_only = -pat.astype(float)
        p_both = np.full(len(y), np.nan)
        for tr, te in GroupKFold(10).split(X, y, grp):
            p_both[te] = LogisticRegression(C=1e6, max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        gain = [auc(y[b], p_both[b]) - auc(y[b], p_only[b]) for b in boot_people(grp) if len(np.unique(y[b])) == 2]
        # (c) paired difference deep - patient status
        diff = [auc(y[b], g.pred.to_numpy()[b]) - auc(y[b], p_only[b]) for b in boot_people(grp)
                if len(np.unique(y[b])) == 2]
        rows += [
            dict(dataset="IRDS", arch=arch, test="odds ratio per SD of model score, adjusted for patient status",
                 value="%.2f [%.2f, %.2f]" % (np.exp(coef), *or_ci)),
            dict(dataset="IRDS", arch=arch, test="AUC patient status only", value=fmt(auc(y, p_only), cluster_ci(auc, y, p_only, grp, n=N_BOOT))),
            dict(dataset="IRDS", arch=arch, test="AUC patient status + model score (grouped 10-fold)",
                 value=fmt(auc(y, p_both), cluster_ci(auc, y, p_both, grp, n=N_BOOT))),
            dict(dataset="IRDS", arch=arch, test="AUC gain of adding the score",
                 value=fmt(auc(y, p_both) - auc(y, p_only), np.percentile(gain, [2.5, 97.5]))),
            dict(dataset="IRDS", arch=arch, test="AUC difference model - patient status (paired)",
                 value=fmt(auc(y, g.pred) - auc(y, p_only), np.percentile(diff, [2.5, 97.5]))),
        ]
        for task, t in g.groupby("task"):
            if t.y.nunique() < 2:
                continue
            rows.append(dict(dataset="IRDS", arch=arch, test="per gesture %d: AUC model / patient status" % task,
                             value="%.2f / %.2f (n=%d, incorrect=%d)" % (auc(t.y, t.pred), auc(t.y, -t.patient.astype(float)),
                                                                        len(t), (t.y == 0).sum())))
    return rows


# ------------------------------------------------------------------ 2. IRDS patient decoding
def irds_patient_decoding():
    import shortcuts as S
    df = S.irds([])
    df = df[df.correct == 1].reset_index(drop=True)
    pred = np.full(len(df), np.nan)
    for gest, g in df.groupby("gesture"):
        idx = g.index.to_numpy()
        pred[idx] = lopo_predict(np.vstack(g.motion), g.patient.to_numpy(), g.subject.to_numpy(), "clf")[0]
    y, grp = df.patient.to_numpy(), df.subject.to_numpy()
    ok = np.isfinite(pred)
    return [dict(dataset="IRDS", arch="logistic regression, movement features",
                 test="decode patient status from CORRECT repetitions only (LOPO per gesture)",
                 value=fmt(auc(y[ok], pred[ok]), cluster_ci(auc, y[ok], pred[ok], grp[ok], n=N_BOOT)) +
                 " (n=%d, patients=%d)" % (ok.sum(), y[ok].sum()))]


# ------------------------------------------------------------------ 3. KIMORE partial rho
def partial_rho(y, p, z):
    ry, rp = rankdata(y), rankdata(p)
    Z = np.column_stack([np.ones(len(z)), z])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    a, b = res(ry), res(rp)
    return float(np.corrcoef(a, b)[0, 1])


def kimore_partial():
    d = pd.read_csv(os.path.join(DEEP, "kimore.csv"))
    d = d[d.protocol == "person"]
    rows = []
    for (task, arch), g in d.groupby(["task", "arch"]):
        y, p, z, grp = g.y.to_numpy(), g.pred.to_numpy(), g.patient.to_numpy().astype(float), g.person.astype(str).to_numpy()
        v = partial_rho(y, p, z)
        bs = [partial_rho(y[b], p[b], z[b]) for b in boot_people(grp)]
        rows.append(dict(dataset="KIMORE", arch=arch, test="Es%d partial Spearman rho, controlling for patient status" % task,
                         value=fmt(v, np.nanpercentile(bs, [2.5, 97.5]))))
    return rows


# ------------------------------------------------------------------ 4. UI-PRMD within-person CI
def uiprmd_within_ci():
    import glob
    import re
    import eval_uiprmd as U
    from shortcuts import uiprmd_rigid
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    recs = []
    for folder, correct in (("Segmented Movements", 1), ("Incorrect Segmented Movements", 0)):
        for p in glob.glob(os.path.join(U.BASE, folder, "Vicon", "Positions", "*.txt")):
            mv, subj, ep = map(int, re.search(r"m(\d+)_s(\d+)_e(\d+)", os.path.basename(p)).groups())
            if mv in U.MOVES:
                recs.append(dict(move=mv, subject=subj, correct=correct, rigid=uiprmd_rigid(p)))
    df = pd.DataFrame(recs)
    rows = []
    rng = np.random.default_rng(0)
    for mv, g in df.groupby("move"):
        per = []
        for s, h in g.groupby("subject"):
            X, y = np.vstack(h.rigid), h.correct.to_numpy()
            if len(set(y)) < 2:
                continue
            pr = np.empty(len(y))
            for i in range(len(y)):
                tr = np.arange(len(y)) != i
                m = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
                pr[i] = m.fit(X[tr], y[tr]).predict_proba(X[i:i + 1])[0, 1]
            per.append(auc(y, pr))
        per = np.array(per)
        bs = [rng.choice(per, len(per)).mean() for _ in range(N_BOOT)]
        rows.append(dict(dataset="UI-PRMD", arch="logistic regression", test="m%02d within-person rigid-spacing AUC (mean over people)" % mv,
                         value=fmt(per.mean(), np.percentile(bs, [2.5, 97.5])) + " (people=%d)" % len(per)))
    return rows


# ------------------------------------------------------------------ 5. REHAB26 random split
def rehab26_random():
    p = os.path.join(DEEP, "rehab26_random.csv")
    if not os.path.exists(p):
        return []
    d = pd.read_csv(p)
    return [dict(dataset="REHAB26", arch=arch, test="Ex%s random split AUC" % task,
                 value=fmt(auc(g.y, g.pred), cluster_ci(auc, g.y.to_numpy(), g.pred.to_numpy(), g.person.astype(str).to_numpy(), n=N_BOOT)))
            for (task, arch), g in d.groupby(["task", "arch"])]


def main(which):
    jobs = dict(irds=irds_conditional, decode=irds_patient_decoding, kimore=kimore_partial,
                uiprmd=uiprmd_within_ci, rehab26=rehab26_random)
    for name in which or list(jobs):
        rows = jobs[name]()
        res = pd.DataFrame(rows)
        res.to_csv(os.path.join(OUT, "review_%s.csv" % name), index=False)
        print(res.to_string(index=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
