"""Run MediaPipe Pose (model_complexity=1) on the REHAB24-6 RGB videos (camera 17).

Frames are resized to 640 px width (close to a 640x480 camera). Saves per
video: world landmarks (frames, 33, 3), visibility (frames, 33), inference ms.
Usage: python run_mediapipe.py [exercise ...]   (default: 1 2)
"""
import glob
import os
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

import rehab24 as R
from paths import DATA, OUT  # noqa: E402

MP_OUT = os.path.join(OUT, "mediapipe")


def run(video, out_path, width=640, complexity=1):
    cap = cv2.VideoCapture(video)
    pose = mp.solutions.pose.Pose(model_complexity=complexity, smooth_landmarks=True)
    world, vis, ms = [], [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (width, int(h * width / w)))
        t0 = time.time()
        res = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        ms.append((time.time() - t0) * 1000)
        if res.pose_world_landmarks:
            lm = res.pose_world_landmarks.landmark
            world.append([[p.x, p.y, p.z] for p in lm])
            vis.append([p.visibility for p in lm])
        else:
            world.append(np.full((33, 3), np.nan))
            vis.append(np.zeros(33))
    pose.close()
    np.savez_compressed(out_path, world=np.array(world, dtype=np.float32),
                        vis=np.array(vis, dtype=np.float32), ms=np.array(ms, dtype=np.float32))
    return len(world), float(np.mean(ms))


def main():
    exercises = [int(x) for x in sys.argv[1:]] or [1, 2]
    os.makedirs(MP_OUT, exist_ok=True)
    for ex in exercises:
        for v in sorted(glob.glob(os.path.join(R.EX, "Ex%d" % ex, "*-Camera17-30fps.mp4"))):
            vid = os.path.basename(v).split("-")[0]
            out = os.path.join(MP_OUT, "Ex%d_%s_c17.npz" % (ex, vid))
            if os.path.exists(out):
                continue
            n, m = run(v, out)
            print("Ex%d %s: %d frames, %.0f ms/frame" % (ex, vid, n, m), flush=True)


if __name__ == "__main__":
    main()
