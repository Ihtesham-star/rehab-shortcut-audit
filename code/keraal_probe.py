"""KERAAL group probe: does the recording group alone (patient / healthy / healthy simulating errors) predict the
binary label? One-hot group as the only input to a logistic regression, leave-one-person-out (strict.lopo_predict),
AUC with person-level cluster bootstrap CI.

Output: results/shortcuts_keraal.csv
"""
import os
import sys

import numpy as np
import pandas as pd

from paths import OUT
from strict import auc, cluster_ci, lopo_predict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep"))
import data as D  # noqa: E402


def main():
    items = D.rehabpile("KERAAL_clf_bn", ["CTK", "ELK", "RTK"], "C")
    d = pd.DataFrame([dict(task=it["task"], group=it["group"], y=it["y"], person=it["person"]) for it in items])
    rows = []
    for task, g in d.groupby("task"):
        X = pd.get_dummies(g.group).to_numpy(float)
        y, grp = g.y.to_numpy(), g.person.to_numpy()
        pred, _ = lopo_predict(X, y, grp, "clf")
        ci = cluster_ci(auc, y, pred, grp)
        rows.append(dict(task=task, model="group label only", n=len(y), people=len(set(grp)),
                         AUC="%.2f [%.2f-%.2f]" % (auc(y, pred), ci[0], ci[1])))
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(OUT, "shortcuts_keraal.csv"), index=False)


if __name__ == "__main__":
    main()
