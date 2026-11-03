"""Skeleton sequences for the deep baselines (paper 0, shortcut audit).

Every loader returns a list of dicts:
  x      (frames, joints, 3) float array in metres-ish, NaN allowed
  y      label (1 = correct / clinician score for KIMORE)
  person grouping id
  task   exercise / movement id (one model per task, or pooled with a task id)
  + dataset-specific metadata used for the shortcut analyses.
Preprocessing (prep): root-centred per frame, scaled by median torso length, NaN filled by
linear interpolation, resampled to T frames.
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))  # code/
from paths import DATA, OUT  # noqa: E402


def _interp(a):
    """Fill NaN frames per coordinate by linear interpolation (edge values held)."""
    a = a.copy()
    f = a.reshape(len(a), -1)
    t = np.arange(len(a))
    for j in range(f.shape[1]):
        ok = np.isfinite(f[:, j])
        f[:, j] = np.interp(t, t[ok], f[ok, j]) if ok.any() else 0.0
    return f.reshape(a.shape)


def prep(x, root, top, T=64):
    """root/top: joint index lists averaged to get the root and the neck/shoulder centre."""
    x = _interp(np.asarray(x, float))
    r = x[:, root].mean(axis=1, keepdims=True)
    tor = np.median(np.linalg.norm(x[:, top].mean(axis=1) - r[:, 0], axis=1))
    x = (x - r) / (tor if tor > 1e-6 else 1.0)
    idx = np.linspace(0, len(x) - 1, T)
    lo, hi = np.floor(idx).astype(int), np.ceil(idx).astype(int)
    w = (idx - lo)[:, None, None]
    return ((1 - w) * x[lo] + w * x[hi]).astype(np.float32)


# MediaPipe 33: hips 23/24, shoulders 11/12. Kinect v2 25: SpineBase 0, SpineShoulder 20.
MP = dict(root=[23, 24], top=[11, 12])
KV2 = dict(root=[0], top=[20])


def rehab24(source="cam", exercises=(1, 2)):
    import rehab24 as R
    seg = R.segmentation()
    seg = seg[seg.exercise_id.isin(exercises) & (seg.mocap_erroneous == 0)]
    out = []
    for (ex, vid), reps in seg.groupby(["exercise_id", "video_id"]):
        if source == "cam":
            w = np.load(os.path.join(OUT, "mediapipe", "Ex%d_%s_c17.npz" % (ex, vid)))["world"].astype(float)
            spec = MP
        else:
            w = R.mocap3d(ex, vid)[..., :3].astype(float)
            spec = dict(root=[R.J["hips"]], top=[R.J["neck"]])
        order = reps.repetition_number.rank().to_numpy()
        for (_, r), pos in zip(reps.iterrows(), order):
            out.append(dict(x=w[r.first_frame:r.last_frame + 1], y=int(r.correctness), person=int(r.person_id),
                            task=int(ex), rep_position=(pos - 1) / max(len(reps) - 1, 1), spec=spec))
    return out


def rehab26(context=False):
    base = os.path.join(DATA, "REHAB26-ViewAngles")
    s = pd.read_csv(os.path.join(base, "Segmentation.csv"), sep=";").sort_values(["video_id", "person_orientation", "first_frame"])
    cache, out = {}, []
    for (vid, o), g in s.groupby(["video_id", "person_orientation"]):
        ex = int(g.exercise_id.iloc[0])
        cam = 1 + int(np.argmin([abs(-o), abs(45 - o), abs(90 - o)]))
        k = (vid, cam)
        if k not in cache:
            cache.clear()
            cache[k] = np.load(os.path.join(base, "extracted", "mediapipe", "ex%d" % ex, "%s_cam%d_3d.npy" % (vid, cam)))[..., :3]
        w = cache[k]
        rows = list(g.itertuples())
        for i, r in enumerate(rows):
            f0, f1 = r.first_frame, r.last_frame
            if context:   # window from the previous rep's start to the next rep's end (same set)
                f0 = rows[i - 1].first_frame if i > 0 else f0
                f1 = rows[i + 1].last_frame if i + 1 < len(rows) else f1
            out.append(dict(x=w[f0:f1 + 1], y=int(r.correctness_score == 1), person=r.person, task=ex,
                            rep_number=r.repetition_number, spec=MP))
    return out


def irds():
    base = os.path.join(DATA, "IntelliRehabDS", "extracted", "SkeletonData", "Simplified")
    out = []
    for p in sorted(glob.glob(os.path.join(base, "*.txt"))):
        s, d, gest, rep, lab, pos = os.path.basename(p)[:-4].split("_")
        if lab not in ("1", "2"):
            continue
        v = np.loadtxt(p, delimiter=",", ndmin=2)
        if v.shape[1] < 75 or len(v) < 16:
            continue
        out.append(dict(x=v[:, :75].reshape(len(v), 25, 3), y=int(lab == "1"), person=s, task=int(gest),
                        patient=int(s.startswith("2")), spec=KV2))
    return out


UIP_LOWER = ["LASI", "RASI", "LPSI", "RPSI", "LTHI", "LKNE", "LTIB", "LANK", "LHEE", "LTOE",
             "RTHI", "RKNE", "RTIB", "RANK", "RHEE", "RTOE"]


def uiprmd(lower_only=False):
    """Shoulder movements m07-m10, Vicon markers. lower_only: pelvis + leg markers only
    (they take no part in a shoulder exercise) -> a movement-free input."""
    import eval_uiprmd as U
    sel = [U.I[m] for m in UIP_LOWER] if lower_only else list(range(39))
    out = []
    for folder, correct in (("Segmented Movements", 1), ("Incorrect Segmented Movements", 0)):
        for p in sorted(glob.glob(os.path.join(U.BASE, folder, "Vicon", "Positions", "*.txt"))):
            mv, subj, ep = map(int, re.search(r"m(\d+)_s(\d+)_e(\d+)", os.path.basename(p)).groups())
            if mv not in U.MOVES:
                continue
            a = np.loadtxt(p, delimiter=",").reshape(-1, 39, 3) / 1000.0
            pel = [UIP_LOWER.index(m) for m in ("LASI", "RASI", "LPSI", "RPSI")] if lower_only else [U.I[m] for m in ("LASI", "RASI", "LPSI", "RPSI")]
            top = [UIP_LOWER.index(m) for m in ("LKNE", "RKNE")] if lower_only else [U.I["C7"], U.I["CLAV"]]
            out.append(dict(x=a[:, sel], y=correct, person=subj, task=mv, episode=ep, spec=dict(root=pel, top=top)))
    return out


def kimore():
    import eval_kimore as K
    out = []
    for subj in glob.glob(os.path.join(K.BASE, "*", "*", "*")):
        group = os.path.relpath(subj, K.BASE).split(os.sep)[1]
        ts = K.clinical(subj)
        if not ts:
            continue
        for ex in K.EX:
            jp = glob.glob(os.path.join(subj, "Es%d" % ex, "Raw", "JointPosition*.csv"))
            if not jp:
                continue
            takes = sorted(p for p in jp if "(1)" not in os.path.basename(p)) or jp
            a = K.load_positions(takes[-1])
            if np.isfinite(a).all(axis=(1, 2)).sum() < 60 or ex not in ts or not np.isfinite(ts[ex]):
                continue
            out.append(dict(x=a, y=float(ts[ex]), person=os.path.basename(subj), task=ex,
                            patient=int(group not in ("Expert", "NotExpert")), spec=KV2))
    return out


RP = os.path.join(DATA, "RehabPile")


def rehabpile(ds, exercises, correct_label):
    """Rehab-Pile archive (fold0 train+test = all recordings). x stored as (n, joints*dims, T), joint-major
    (DeepRehabPile utils); the first 3 dims are positions (2-D datasets get z = 0). Loaded without pickle.
    KERAAL person ids: R = group 1A patients, S = group 2A healthy, P = group 3 healthy simulating errors
    (inferred from the raw KERAAL file names and counts)."""
    import json
    out = []
    for ex in exercises:
        d = os.path.join(RP, ds, ex)
        info = json.load(open(os.path.join(d, "info.json")))
        J, Dm = info["n_joints"], info.get("n_dim", info.get("dim"))
        L = lambda n: np.load(os.path.join(d, "fold0", n), allow_pickle=False)
        x = np.concatenate([L("x_train_fold0.npy"), L("x_test_fold0.npy")])
        y = np.concatenate([L("y_train_fold0.npy"), L("y_test_fold0.npy")]).astype(str)
        s = np.concatenate([L("s_train_fold0.npy"), L("s_test_fold0.npy")]).astype(str)
        x = x.reshape(len(x), J, Dm, -1)[:, :, :3].transpose(0, 3, 1, 2)          # (n, T, J, <=3)
        if x.shape[-1] < 3:
            x = np.concatenate([x, np.zeros(x.shape[:-1] + (3 - x.shape[-1],), x.dtype)], -1)
        tree = info["kinematic_tree"]
        spec = dict(root=[tree[0][0]], top=[tree[1][0]]) if ds.startswith("KERAAL") else dict(root=list(range(J)), top=[0])
        for i in range(len(x)):
            out.append(dict(x=x[i], y=int(y[i] == correct_label), person=s[i], task=ex, group=s[i][0], spec=spec))
    return out


def tensor(items, T=64):
    X = np.stack([prep(it["x"], it["spec"]["root"], it["spec"]["top"], T) for it in items])
    return X
