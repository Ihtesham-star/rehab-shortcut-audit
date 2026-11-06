"""Deep baselines (LSTM, ST-GCN) under different evaluation protocols; saves out-of-fold predictions.

  python deep/run.py EXPERIMENT [EXPERIMENT ...] [--epochs N] [--seeds N]

Protocols:
  person : grouped CV, each person in exactly one test fold (LOPO if <= 10 people, else 10 folds)
  random : stratified 5-fold over recordings/reps, people mixed between train and test (common in the literature)
One model per task (exercise). Predictions = mean over seeds. Normalisation and target scaling
are fitted on the training fold only; fixed epochs (no test-fold early stopping).
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import data as D  # noqa: E402
from models import LSTM, STGCN, knn_adjacency  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/
from paths import DATA, OUT  # noqa: E402

DEEP_OUT = os.path.join(OUT, "deep")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fit_predict(Xtr, ytr, Xte, kind, arch, epochs, seed, wtr=None, gtr=None, eta=0.01):
    """wtr: optional per-sample loss weights (mean 1).
    gtr: optional group id per sample (0..G-1). If given, train with online group DRO (Sagawa et al., 2020):
    keep a weight q_g per group, raise it for groups with high loss (q_g <- q_g * exp(eta * loss_g)) and
    minimise the q-weighted group loss, so the worst group gets most attention."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    J = Xtr.shape[2]
    model = LSTM(J) if arch == "lstm" else STGCN(J, knn_adjacency(Xtr.mean((0, 1))))
    model = model.to(DEV)
    mu, sd = (ytr.mean(), ytr.std() + 1e-6) if kind == "reg" else (0.0, 1.0)
    yt = torch.tensor((ytr - mu) / sd, dtype=torch.float32)
    Xt = torch.tensor(Xtr)
    wt = torch.tensor(np.ones(len(Xtr)) if wtr is None else wtr, dtype=torch.float32)
    if gtr is not None:
        gt = torch.tensor(gtr, dtype=torch.long)
        G = int(gt.max()) + 1
        q = torch.ones(G, device=DEV) / G
    if kind == "clf":
        pos = max(ytr.mean(), 1e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor((1 - pos) / pos, device=DEV), reduction="none")
    else:
        loss_fn = nn.MSELoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    bs = 32
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            xb = Xt[b].to(DEV)
            xb = xb * torch.empty(len(b), 1, 1, 1, device=DEV).uniform_(0.9, 1.1) + 0.01 * torch.randn_like(xb)
            per = loss_fn(model(xb), yt[b].to(DEV)) * wt[b].to(DEV)
            if gtr is None:
                loss = per.mean()
            else:
                gb = gt[b].to(DEV)
                present = torch.unique(gb)
                gl = torch.stack([per[gb == g].mean() for g in present])
                with torch.no_grad():
                    q[present] = q[present] * torch.exp(eta * gl.detach())
                    q /= q.sum()
                loss = (q[present] * gl).sum() / q[present].sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    model.eval()
    with torch.no_grad():
        p = torch.cat([model(torch.tensor(Xte[i:i + 256]).to(DEV)).cpu() for i in range(0, len(Xte), 256)]).numpy()
    return 1 / (1 + np.exp(-p)) if kind == "clf" else p * sd + mu


def group_balance(y, patient):
    """Mitigation (review): inside the CORRECT class, give patients and healthy people the same total weight,
    so that 'patient' is no longer a cheap cue for 'incorrect'. Incorrect reps keep weight 1 (almost all are
    from patients). Weights are computed on the training fold only and rescaled to mean 1."""
    w = np.ones(len(y))
    c = y == 1
    for grp in (0, 1):
        m = c & (patient == grp)
        if m.any():
            w[m] = c.sum() / (2.0 * m.sum())
    return w / w.mean()


def reweigh(y, patient):
    """Reweighing (Kamiran and Calders, 2012): w(g, y) = P(g) P(y) / P(g, y), computed on the training fold, so that
    in the weighted data the label is independent of patient status. Rescaled to mean 1."""
    w = np.ones(len(y))
    for g in (0, 1):
        for c in (0, 1):
            m = (patient == g) & (y == c)
            if m.any():
                w[m] = (patient == g).mean() * (y == c).mean() / m.mean()
    return w / w.mean()


def folds(y, groups, protocol, kind, seed=0):
    n_people = len(np.unique(groups))
    if protocol == "person":
        return GroupKFold(n_splits=min(n_people, 10)).split(y, y, groups)
    if kind == "clf":
        return StratifiedKFold(5, shuffle=True, random_state=seed).split(y, y)
    return KFold(5, shuffle=True, random_state=seed).split(y)


def run(name, items, kind, protocols, archs, epochs, seeds, T=64, train_mask=None):
    """train_mask: optional boolean array; if given, only those items are used for training (all are predicted
    only if they are in a test fold AND in the mask's complement handling below: we simply restrict to the mask)."""
    rows = []
    tasks = sorted(set(it["task"] for it in items))
    for task in tasks:
        sub = [it for it in items if it["task"] == task]
        X = D.tensor(sub, T)
        y = np.array([it["y"] for it in sub], float if kind == "reg" else int)
        g = np.array([str(it["person"]) for it in sub])
        meta = pd.DataFrame([{k: v for k, v in it.items() if k not in ("x", "spec")} for it in sub])
        for protocol in protocols:
            for arch in archs:
                t0 = time.time()
                pred = np.zeros(len(y))
                for tr, te in folds(y, g, protocol, kind):
                    if kind == "clf" and len(np.unique(y[tr])) < 2:
                        pred[te] = np.nan
                        continue
                    w, gid = None, None
                    if name.endswith("_gdro"):  # groups = patient status x label
                        gid = 2 * meta.patient.to_numpy()[tr] + y[tr]
                    if name.endswith("_balanced"):
                        w = group_balance(y[tr], meta.patient.to_numpy()[tr])
                    elif name.endswith("_reweigh"):
                        w = reweigh(y[tr], meta.patient.to_numpy()[tr])
                    pred[te] = np.mean([fit_predict(X[tr], y[tr], X[te], kind, arch, epochs, s, w, gid) for s in range(seeds)], axis=0)
                m = meta.copy()
                m["pred"], m["protocol"], m["arch"], m["experiment"] = pred, protocol, arch, name
                rows.append(m)
                print("%-14s task %-3s %-6s %-5s %.0fs" % (name, task, protocol, arch, time.time() - t0), flush=True)
    res = pd.concat(rows, ignore_index=True)
    os.makedirs(DEEP_OUT, exist_ok=True)
    res.to_csv(os.path.join(DEEP_OUT, name + ".csv"), index=False)
    return res


EXPERIMENTS = {
    # name: (loader, kind, protocols, T)
    "rehab24_cam": (lambda: D.rehab24("cam", (1, 2)), "clf", ("person", "random"), 64),
    "rehab24_mocap": (lambda: D.rehab24("mocap", (1, 2, 3, 4, 5, 6)), "clf", ("person", "random"), 64),
    "rehab26_single": (lambda: D.rehab26(False), "clf", ("person",), 64),
    "rehab26_context": (lambda: D.rehab26(True), "clf", ("person",), 128),
    "rehab26_random": (lambda: D.rehab26(False), "clf", ("random",), 64),  # added for review: random split
    "irds": (D.irds, "clf", ("person", "random"), 64),
    "irds_balanced": (D.irds, "clf", ("person",), 64),  # review: group-balanced training (mitigation)
    "irds_reweigh": (D.irds, "clf", ("person",), 64),  # Kamiran-Calders reweighing (label independent of group)
    "irds_gdro": (D.irds, "clf", ("person",), 64),  # group DRO over patient status x label
    "irds_patients": (lambda: [it for it in D.irds() if it["patient"] == 1], "clf", ("person",), 64),
    "uiprmd_full": (lambda: D.uiprmd(False), "clf", ("person", "random"), 64),
    "uiprmd_lower": (lambda: D.uiprmd(True), "clf", ("person", "random"), 64),
    "kimore": (D.kimore, "reg", ("person", "random"), 128),
    "keraal": (lambda: D.rehabpile("KERAAL_clf_bn", ["CTK", "ELK", "RTK"], "C"), "clf", ("person", "random"), 64),
    "keraal_patients": (lambda: [it for it in D.rehabpile("KERAAL_clf_bn", ["CTK", "ELK", "RTK"], "C") if it["group"] == "R"],
                        "clf", ("person",), 64),
    "ucdhe": (lambda: D.rehabpile("UCDHE_clf_bn", ["MP", "Rowing"], "n"), "clf", ("person", "random"), 64),
}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--archs", default="lstm,stgcn")
    a = ap.parse_args()
    for e in (EXPERIMENTS if a.experiments == ["all"] else a.experiments):
        loader, kind, protocols, T = EXPERIMENTS[e]
        t0 = time.time()
        items = loader()
        print("== %s: %d items, loaded in %.0fs" % (e, len(items), time.time() - t0), flush=True)
        run(e, items, kind, protocols, a.archs.split(","), a.epochs, a.seeds, T)
