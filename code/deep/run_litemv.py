"""LITEMV (Rehab-Pile benchmark, Ismail-Fawaz et al.) on IntelliRehabDS and KIMORE with the authors' released code.

The model classes come unchanged from github.com/MSD-IRIMAS/DeepRehabPile (deep_rehab_pile.classifiers /
deep_rehab_pile.regressors, LITE_MV_*), with their default settings (32 filters, kernel 40, batch 64,
Adam, learning-rate decay on plateau, best epoch by training loss). Only the data and the folds are ours:
the same skeleton preprocessing and person-wise folds as deep/run.py, so the results can be compared with
the LSTM and ST-GCN directly. Needs TensorFlow (run with the separate environment that has it).

  python deep/run_litemv.py irds kimore [--epochs N] [--runs N] [--tasks ...] [--part SUFFIX]
Run with KERAS_BACKEND=torch to train on the GPU through Keras 3.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RP_CODE = os.environ.get("REHABPILE_CODE", os.path.join(HERE, "..", "..", "external", "DeepRehabPile"))  # git clone of MSD-IRIMAS/DeepRehabPile
sys.path.insert(0, RP_CODE)
import data as D  # noqa: E402
from run import folds  # noqa: E402

sys.path.insert(0, os.path.join(HERE, ".."))
from paths import OUT as _OUT  # noqa: E402
OUT = os.path.join(_OUT, "deep")


def to_rp(X):
    """(n, T, J, 3) -> (n, J*3, T), joint-major channels as in the Rehab-Pile archive."""
    n, T, J, C = X.shape
    return np.ascontiguousarray(X.transpose(0, 2, 3, 1).reshape(n, J * C, T)), J, C


def fit_predict(Xtr, ytr, Xte, kind, epochs, runs):
    from deep_rehab_pile.classifiers import LITE_MV_CLASSIFIER
    from deep_rehab_pile.regressors import LITE_MV_REGRESSOR
    Xtr, J, C = to_rp(Xtr)
    Xte, _, _ = to_rp(Xte)
    mu, sd = (ytr.mean(), ytr.std() + 1e-6) if kind == "reg" else (0.0, 1.0)
    preds = []
    for r in range(runs):
        tmp = tempfile.mkdtemp() + os.sep
        cls = LITE_MV_CLASSIFIER if kind == "clf" else LITE_MV_REGRESSOR
        m = cls(output_dir=tmp, best_file_name="best_%d" % r, init_file_name="init_%d" % r,
                length_TS=Xtr.shape[2], n_joints=J, n_dim=C, epochs=epochs)
        if kind == "clf":
            m.fit(Xtr, ytr)
            preds.append(m.predict_proba(Xte)[:, 1])    # classes sorted: column 1 = label 1 (correct)
        else:
            m.fit(Xtr, (ytr - mu) / sd)
            preds.append(np.asarray(m.predict_proba(Xte) if hasattr(m, "predict_proba") else m.predict(Xte)[0]).ravel() * sd + mu)
        shutil.rmtree(tmp, ignore_errors=True)
    return np.mean(preds, axis=0)


EXP = {"irds": (D.irds, "clf", 64), "kimore": (D.kimore, "reg", 128)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("experiments", nargs="+")
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--tasks", nargs="*", help="only these tasks (to split the work over several processes)")
    ap.add_argument("--part", default="", help="suffix for the output file of this part")
    a = ap.parse_args()
    for name in a.experiments:
        loader, kind, T = EXP[name]
        items = loader()
        rows = []
        for task in sorted(set(it["task"] for it in items)):
            if a.tasks and str(task) not in a.tasks:
                continue
            sub = [it for it in items if it["task"] == task]
            X = D.tensor(sub, T)
            y = np.array([it["y"] for it in sub], float if kind == "reg" else int)
            g = np.array([str(it["person"]) for it in sub])
            meta = pd.DataFrame([{k: v for k, v in it.items() if k not in ("x", "spec")} for it in sub])
            pred, t0 = np.zeros(len(y)), time.time()
            for tr, te in folds(y, g, "person", kind):
                pred[te] = fit_predict(X[tr], y[tr], X[te], kind, a.epochs, a.runs)
            meta["pred"], meta["protocol"], meta["arch"], meta["experiment"] = pred, "person", "litemv", name + "_litemv"
            rows.append(meta)
            print("%s task %s litemv %.0fs" % (name, task, time.time() - t0), flush=True)
        os.makedirs(OUT, exist_ok=True)
        pd.concat(rows, ignore_index=True).to_csv(os.path.join(OUT, name + "_litemv%s.csv" % a.part), index=False)


if __name__ == "__main__":
    main()
