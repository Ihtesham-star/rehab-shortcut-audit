"""REHAB24-6 loading helpers (Zenodo 13305826)."""
import os

import numpy as np
import pandas as pd
from paths import DATA, OUT  # noqa: E402

ROOT = os.path.join(DATA, "REHAB24-6")
EX = os.path.join(ROOT, "extracted")
# joint indices from joints_names.txt
J = {"hips": 0, "neck": 3, "head": 4,
     "l_shoulder": 7, "l_elbow": 8, "l_wrist": 9, "r_shoulder": 12, "r_elbow": 13, "r_wrist": 14,
     "l_hip": 16, "r_hip": 21}


def segmentation():
    return pd.read_csv(os.path.join(ROOT, "Segmentation.csv"), sep=";")


def mocap3d(exercise, video_id):
    """(frames, 26, 3) in metres, y up, 30 fps (frame-aligned with the videos)."""
    a = np.load(os.path.join(EX, "Ex%d" % exercise, "%s-30fps.npy" % video_id))
    return a[:, :, :3]


def video_path(exercise, video_id, cam=17):
    name = "%s-Camera17-30fps.mp4" % video_id if cam == 17 else "%s-Camera18-30fps-transposed.mp4" % video_id
    return os.path.join(EX, "Ex%d" % exercise, name)


def mocap_points(a):
    g = lambda k: a[:, J[k]]
    return dict(ls=g("l_shoulder"), rs=g("r_shoulder"), lh=g("l_hip"), rh=g("r_hip"),
                le=g("l_elbow"), re=g("r_elbow"), lw=g("l_wrist"), rw=g("r_wrist"))
