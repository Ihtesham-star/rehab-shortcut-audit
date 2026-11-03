"""Summarise deep-baseline out-of-fold predictions (out/deep/*.csv) into the shortcut tables."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from strict import auc, cluster_ci, rho  # noqa: E402
from paths import DATA, OUT  # noqa: E402

OUT = os.path.join(OUT, "deep")


def f(metric, y, p, g):
    ok = np.isfinite(p)
    y, p, g = np.asarray(y)[ok], np.asarray(p)[ok], np.asarray(g)[ok]
    ci = cluster_ci(metric, y, p, g, n=1000)
    return "%.2f [%.2f-%.2f]" % (metric(y, p), ci[0], ci[1])


def load(name):
    p = os.path.join(OUT, name + ".csv")
    return pd.read_csv(p) if os.path.exists(p) else None


def main():
    rows = []
    add = lambda **k: rows.append(k)
    for name in ("rehab24_cam", "rehab24_mocap", "uiprmd_full", "uiprmd_lower", "rehab26_single", "rehab26_context",
                 "rehab26_random"):
        d = load(name)
        if d is None:
            continue
        for (task, prot, arch), g in d.groupby(["task", "protocol", "arch"]):
            add(experiment=name, task=task, protocol=prot, arch=arch, subset="all", metric="AUC",
                value=f(auc, g.y, g.pred, g.person))
            if name.startswith("rehab24") and prot == "person":
                for half, m in (("first half of video", g.rep_position <= 0.5), ("second half", g.rep_position > 0.5)):
                    add(experiment=name, task=task, protocol=prot, arch=arch, subset=half, metric="AUC",
                        value=f(auc, g.y[m], g.pred[m], g.person[m]))
    for name in ("irds", "irds_patients", "irds_balanced", "irds_reweigh"):
        d = load(name)
        if d is None:
            continue
        for (prot, arch), g in d.groupby(["protocol", "arch"]):
            add(experiment=name, task="9 gestures pooled", protocol=prot, arch=arch, subset="all", metric="AUC",
                value=f(auc, g.y, g.pred, g.person))
            if name in ("irds", "irds_balanced", "irds_reweigh"):
                m = g.patient == 1
                add(experiment=name, task="9 gestures pooled", protocol=prot, arch=arch, subset="patients only",
                    metric="AUC", value=f(auc, g.y[m], g.pred[m], g.person[m]))
                add(experiment=name, task="9 gestures pooled", protocol=prot, arch="(patient label only)",
                    subset="all", metric="AUC", value=f(auc, g.y, -g.patient.astype(float), g.person))
    d = load("kimore")
    if d is not None:
        for (task, prot, arch), g in d.groupby(["task", "protocol", "arch"]):
            m = g.patient == 1
            add(experiment="kimore", task=task, protocol=prot, arch=arch, subset="all", metric="rho",
                value=f(rho, g.y, g.pred, g.person))
            add(experiment="kimore", task=task, protocol=prot, arch=arch, subset="patients only", metric="rho",
                value=f(rho, g.y[m], g.pred[m], g.person[m]))
            if arch == "lstm" and prot == "person":
                add(experiment="kimore", task=task, protocol=prot, arch="(patient label only)", subset="all",
                    metric="rho", value=f(rho, g.y, -g.patient.astype(float), g.person))
    for name in ("keraal", "keraal_patients", "ucdhe"):
        d = load(name)
        if d is None:
            continue
        for (task, prot, arch), g in d.groupby(["task", "protocol", "arch"]):
            add(experiment=name, task=task, protocol=prot, arch=arch, subset="all", metric="AUC",
                value=f(auc, g.y, g.pred, g.person))
            if name == "keraal":
                for grp, lab in (("R", "patients only (R)"), ("P", "simulating group only (P)")):
                    m = g.group == grp
                    if g.y[m].nunique() == 2:
                        add(experiment=name, task=task, protocol=prot, arch=arch, subset=lab, metric="AUC",
                            value=f(auc, g.y[m], g.pred[m], g.person[m]))
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_rows", 500)
    for e, g in res.groupby("experiment", sort=False):
        print("\n==", e, "==")
        print(g.drop(columns="experiment").to_string(index=False))
    res.to_csv(os.path.join(OUT, "summary.csv"), index=False)


if __name__ == "__main__":
    main()
