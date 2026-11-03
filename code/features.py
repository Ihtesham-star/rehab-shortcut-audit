"""Per-frame movement-quality signals and repetition counting, shared by mocap and camera.

The signals match the mistakes a physiotherapist checks (REHAB26 error list, same team):
- arm forward of the body plane (ERR_ARM_FORWARD)      -> plane_dev
- shoulder shrug / elevation (ERR_ELEVATED_SHOULDER)    -> shrug
- head tilted to the side (ERR_HEAD_TO_SIDE)            -> head_tilt
- arm not raised enough (ERR_LOW_ABDUCTION)             -> elevation (range)
- elbow bent (ERR_BENT_ARM)                             -> elbow
plus trunk lean (compensation), left/right symmetry and smoothness.
"""
import numpy as np


def _unit(v):
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n < 1e-9, 1.0, n)


def body_frame(ls, rs, lh, rh):
    L = _unit(ls - rs)
    U = _unit((ls + rs) / 2 - (lh + rh) / 2)
    U = _unit(U - np.sum(U * L, -1, keepdims=True) * L)
    return L, U, np.cross(L, U)


def signals(p, head_top, head_base):
    """p: dict ls, rs, lh, rh, le, re, lw, rw (N,3). head_top/head_base: (N,3) points
    defining the head axis (mocap Head/Neck; camera mid-ears/mid-shoulders).
    Returns dict of (N,) arrays per side plus shared ones."""
    L, U, F = body_frame(p["ls"], p["rs"], p["lh"], p["rh"])
    torso = np.linalg.norm((p["ls"] + p["rs"]) / 2 - (p["lh"] + p["rh"]) / 2, axis=-1)
    out = {}
    for side, sh, el, wr, hip in (("left", p["ls"], p["le"], p["lw"], p["lh"]),
                                  ("right", p["rs"], p["re"], p["rw"], p["rh"])):
        u = _unit(el - sh)
        f = _unit(wr - el)
        down = -np.sum(u * U, -1)
        out[side] = {
            "elevation": np.degrees(np.arccos(np.clip(down, -1, 1))),
            "plane_dev": np.degrees(np.arcsin(np.clip(np.sum(u * F, -1), -1, 1))),  # + = forward
            "elbow": np.degrees(np.arccos(np.clip(np.sum(u * f, -1), -1, 1))),
            # shoulder height above its hip, in torso lengths (shrug raises it)
            "shoulder_h": np.sum((sh - hip) * U, -1) / np.where(torso < 1e-6, 1, torso),
        }
    h = _unit(head_top - head_base)
    out["head_tilt"] = np.degrees(np.arcsin(np.clip(np.sum(h * L, -1), -1, 1)))
    up0 = _unit(np.nanmean(U[:30], axis=0))
    out["trunk_lean"] = np.degrees(np.arccos(np.clip(U @ up0, -1, 1)))   # vs start of the recording
    out["torso_up"] = U                                                  # for per-rep lean
    return out


def lean_within(torso_up, sl, ref_frames=5):
    """Max trunk tilt (deg) during a rep, relative to the torso at the rep's own start.
    (Measuring against the start of the whole video drifts with rep order; review 2026-10-31.)"""
    U = torso_up[sl]
    ref = _unit(np.nanmean(U[:ref_frames], axis=0))
    return np.nanmax(np.degrees(np.arccos(np.clip(U @ ref, -1, 1))))


def count_reps(x, fps=30.0, lo_frac=0.35, hi_frac=0.65, min_s=0.6):
    """Count repetitions with hysteresis on a signal that rises during each rep.
    Thresholds are relative to this recording's own range (robust percentiles of the whole
    recording, so this is an offline count). A rep must stay above the high threshold for
    at least min_s * 0.5 seconds."""
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    if ok.sum() < 10:
        return 0, []
    lo_v, hi_v = np.percentile(x[ok], 5), np.percentile(x[ok], 95)
    lo, hi = lo_v + lo_frac * (hi_v - lo_v), lo_v + hi_frac * (hi_v - lo_v)
    state, start, reps = "low", None, []
    for i, v in enumerate(x):
        if not np.isfinite(v):
            continue
        if state == "low" and v > hi:
            state, start = "high", i
        elif state == "high" and v < lo:
            if (i - start) / fps >= min_s * 0.5:
                reps.append((start, i))
            state = "low"
    return len(reps), reps


def sparc(v, fs=30.0, fc=10.0):
    """Spectral arc length of the speed profile (closer to 0 = smoother)."""
    v = np.asarray(v, float)
    ok = np.isfinite(v)
    if ok.sum() < 8:
        return np.nan
    idx = np.arange(len(v))
    v = np.interp(idx, idx[ok], v[ok])
    s = np.abs(np.gradient(v)) * fs
    if s.max() <= 0:
        return np.nan
    nfft = int(2 ** (np.ceil(np.log2(len(s))) + 4))
    f = np.arange(nfft) * fs / nfft
    S = np.abs(np.fft.fft(s, nfft))
    S = S / S.max()
    sel = f <= fc
    return -np.sum(np.sqrt((np.diff(f[sel]) / fc) ** 2 + np.diff(S[sel]) ** 2))
