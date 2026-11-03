"""REHAB26-ViewAngles: label-design shortcut and whether it is exploitable without movement.

Per repetition, the camera with the most frontal view is used (view angle cam1=-o, cam2=45-o, cam3=90-o).
Models (all LOPO, strict.lopo_predict):
- rep number (categorical)                       -> label design only
- REST features: 10 frames before the rep starts, LOWER BODY only (arm exercises 1 and 7),
  plus where/how big the person is in the image -> no exercise movement involved
- movement features of the rep itself (generic: joint ranges and speeds, hip-centred, torso-scaled)
"""
import os
import sys

import numpy as np
import pandas as pd

from strict import auc, cluster_ci, lopo_predict
from paths import DATA, OUT  # noqa: E402

BASE = os.path.join(DATA, "REHAB26-ViewAngles")
LOWER = [23, 24, 25, 26, 27, 28, 29, 30, 31, 32]
BODY = list(range(11, 33))
_cache = {}


def arr(vid, ex, cam, kind):
    k = (vid, cam, kind)
    if k not in _cache:
        _cache[k] = np.load(os.path.join(BASE, "extracted", "mediapipe", "ex%d" % ex, "%s_cam%d_%s.npy" % (vid, cam, kind)))
    return _cache[k]


def centred(a):
    hip = (a[:, 23] + a[:, 24]) / 2
    sh = (a[:, 11] + a[:, 12]) / 2
    torso = np.nanmedian(np.linalg.norm(sh - hip, axis=1)) or 1.0
    return (a - hip[:, None]) / torso


def rest_features(w3, w2, f0):
    lo = max(f0 - 10, 0)
    if f0 - lo < 3:
        return np.full(len(LOWER) * 3 + 3, np.nan)
    p = centred(w3[lo:f0, :, :3])[:, LOWER].mean(axis=0).ravel()
    img = w2[lo:f0, :, :2]
    hip = ((img[:, 23] + img[:, 24]) / 2).mean(axis=0)
    size = np.linalg.norm(((img[:, 11] + img[:, 12]) / 2 - (img[:, 23] + img[:, 24]) / 2), axis=1).mean()
    return np.concatenate([p, hip, [size]])


def motion_features(w3, f0, f1):
    a = centred(w3[f0:f1 + 1, :, :3])[:, BODY]
    rng = np.nanpercentile(a, 95, axis=0) - np.nanpercentile(a, 5, axis=0)
    spd = np.nanmean(np.abs(np.diff(a, axis=0)), axis=0)
    return np.concatenate([rng.ravel(), spd.ravel()])


def main():
    s = pd.read_csv(os.path.join(BASE, "Segmentation.csv"), sep=";")
    s["correct"] = (s.correctness_score == 1).astype(int)
    rows = []
    for ex, g in s.groupby("exercise_id"):
        rest, mot = [], []
        for _, r in g.iterrows():
            o = r.person_orientation
            cam = 1 + int(np.argmin([abs(-o), abs(45 - o), abs(90 - o)]))
            w3, w2 = arr(r.video_id, ex, cam, "3d"), arr(r.video_id, ex, cam, "2d")
            rest.append(rest_features(w3, w2, r.first_frame))
            mot.append(motion_features(w3, r.first_frame, r.last_frame))
        y, grp = g.correct.to_numpy(), g.person.to_numpy()
        models = {"rep number ONLY (label design)": pd.get_dummies(g.repetition_number.clip(upper=12)).to_numpy(float),
                  "movement of the rep (generic features)": np.vstack(mot)}
        if ex in (1, 7):
            models["REST before rep, lower body + image position ONLY"] = np.vstack(rest)
        for name, X in models.items():
            p, _ = lopo_predict(X, y, grp, "clf")
            ci = cluster_ci(auc, y, p, grp)
            rows.append(dict(task="Ex%d" % ex, model=name, n=len(y), people=len(set(grp)),
                             AUC="%.2f [%.2f-%.2f]" % (auc(y, p), ci[0], ci[1])))
            print(rows[-1], flush=True)
    res = pd.DataFrame(rows)
    print(res.to_string(index=False))
    res.to_csv(os.path.join(OUT, "shortcuts_rehab26.csv"), index=False)


if __name__ == "__main__":
    main()
