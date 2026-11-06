"""Paper 0 figures, journal style. Reads results/, writes results/figures/figN.{pdf,png}.

Style: no titles inside figures (captions live in figure_captions.md), bold panel letters,
Arial 7 pt, widths 88 mm (single column) / 180 mm (double column), vector PDF + 600 dpi PNG.
  python make_figures.py
"""
import os
import sys
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/
from paths import DATA, OUT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)
os.makedirs(FIG, exist_ok=True)
MM = 1 / 25.4
W1, W2 = 88 * MM, 180 * MM

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY, INK, INK2, GRID = "#8f8e88", "#0b0b0b", "#52514e", "#ebeae5"
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "font.size": 7,
    "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.axisbelow": True, "legend.frameon": False, "legend.handlelength": 1.2, "legend.handletextpad": 0.4,
    "legend.columnspacing": 1.0, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def parse(v):
    """'0.84 [0.75-0.91]', 'rho -0.08 [-0.42-0.29]' -> (value, low, high). The '-' between low and high is a separator."""
    m = re.search(r"(-?\d+\.\d+)\s*\[\s*(-?\d+\.\d+)\s*-\s*(-?\d+\.\d+)\s*\]", str(v))
    if m:
        return tuple(float(g) for g in m.groups())
    return (float(re.search(r"-?\d+\.\d+", str(v)).group()), np.nan, np.nan)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=600)
    plt.close(fig)
    print("saved", name)


def letter(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="left")


def chance(ax, v=0.5):
    ax.axhline(v, color=INK2, lw=0.6, ls=(0, (2, 2)), zorder=1)


S = pd.read_csv(os.path.join(OUT, "deep", "summary.csv"))


def get(exp, task=None, prot="person", arch="stgcn", subset="all"):
    m = (S.experiment == exp) & (S.protocol == prot) & (S.arch == arch) & (S.subset == subset)
    if task is not None:
        m &= S.task.astype(str) == str(task)
    r = S[m]
    return parse(r.value.iloc[0]) if len(r) else (np.nan,) * 3


def whisker(ax, x, est, color, marker="o", size=18, hollow=False, lw=0.9):
    """Point estimate with its 95% CI; a CI that misses the estimate (very few people) is widened to include it."""
    v, lo, hi = est
    lo, hi = (min(lo, v), max(hi, v)) if np.isfinite(lo) else (v, v)
    ax.plot([x, x], [lo, hi], color=color, lw=lw, solid_capstyle="round", zorder=2)
    ax.scatter(x, v, s=size, marker=marker, color="white" if hollow else color, edgecolor=color, lw=0.9, zorder=3)


# ---------------------------------------------------------------- Fig 1
def fig1():
    spec = [("REHAB24-6 (RGB)", "rehab24_cam", {1: "Arm abduction", 2: "Arm V-W"}),
            ("REHAB26-ViewAngles", ("rehab26_single", "rehab26_random"), {1: "Arm abduction", 6: "Squat", 7: "Shoulder extension"}),
            ("REHAB24-6 (mocap)", "rehab24_mocap", {i: "Exercise %d" % i for i in range(1, 7)}),
            ("UI-PRMD", "uiprmd_full", {7: "Abduction", 8: "Extension", 9: "Int./ext. rotation", 10: "Scaption"}),
            ("IntelliRehabDS", "irds", {"9 gestures pooled": "9 gestures"}),
            ("KERAAL", "keraal", {"CTK": "CTK", "ELK": "ELK", "RTK": "RTK"}),
            ("UCDHE", "ucdhe", {"MP": "Military press", "Rowing": "Rowing"})]
    fig, ax = plt.subplots(figsize=(W1, 124 * MM))
    y, yt, yl, heads, rows = 0, [], [], [], []
    for ds, exp, tasks in spec:
        heads.append((y, ds))
        y += 0.9
        for t, lab in tasks.items():
            ep, er = exp if isinstance(exp, tuple) else (exp, exp)
            p = np.nanmean([get(ep, t, "person", a)[0] for a in ("lstm", "stgcn")])
            r = np.nanmean([get(er, t, "random", a)[0] for a in ("lstm", "stgcn")])
            ax.plot([p, r], [y, y], color="#cfcec8", lw=2.2, solid_capstyle="round", zorder=1)
            ax.scatter(p, y, s=16, color=BLUE, zorder=3, lw=0)
            ax.scatter(r, y, s=16, color=ORANGE, zorder=3, lw=0)
            ax.text(1.02, y, "+%.2f" % (r - p), va="center", ha="left", fontsize=6, color=INK2, transform=ax.get_yaxis_transform())
            yt.append(y)
            yl.append(lab)
            rows.append((ds, lab, round(p, 3), round(r, 3)))
            y += 1
        y += 0.3
    ax.set_yticks(yt, yl)
    for yy, ds in heads:
        ax.text(-0.02, yy, ds, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=6.8, fontweight="bold")
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(y - 0.1, -0.7)
    ax.set_xlim(0.5, 1.0)
    ax.grid(axis="y", visible=False)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("AUC (mean of LSTM and ST-GCN)")
    ax.text(1.02, -0.7, "Δ", transform=ax.get_yaxis_transform(), ha="left", va="center", fontsize=6.5, color=INK2)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="Person-wise split"),
                       Line2D([], [], marker="o", ls="", color=ORANGE, ms=4, label="Random split")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2)
    save(fig, "fig1_person_vs_random")
    pd.DataFrame(rows, columns=["dataset", "task", "person_split", "random_split"]).to_csv(os.path.join(FIG, "fig1_data.csv"), index=False)


# ---------------------------------------------------------------- Fig 2
def fig2():
    fig = plt.figure(figsize=(W2, 58 * MM))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.45, 1.05, 1.15], wspace=0.36)
    # a: IntelliRehabDS
    ax = fig.add_subplot(gs[0])
    # color = which people the model was tested on; hollow = trained on patients only (no brackets needed)
    items = [(0, get("irds", None, "person", "(patient label only)"), GRAY, "D", False),
             (1.4, get("irds", None, "person", "lstm"), BLUE, "o", False), (2.4, get("irds", None, "person", "stgcn"), BLUE, "o", False),
             (3.4, get("irds_litemv", None, "person", "litemv"), BLUE, "o", False),
             (4.8, get("irds", None, "person", "lstm", "patients only"), AQUA, "o", False),
             (5.8, get("irds", None, "person", "stgcn", "patients only"), AQUA, "o", False),
             (6.8, get("irds_litemv", None, "person", "litemv", "patients only"), AQUA, "o", False),
             (8.2, get("irds_patients", None, "person", "lstm"), AQUA, "o", True),
             (9.2, get("irds_patients", None, "person", "stgcn"), AQUA, "o", True)]
    for x, e, c, m, hol in items:
        whisker(ax, x, e, c, m, size=20, hollow=hol)
    for xs in (0.7, 4.1, 7.5):  # light separators between the groups
        ax.axvline(xs, color="#e2e1dc", lw=0.6, zorder=0)
    ax.set_xticks([0, 1.4, 2.4, 3.4, 4.8, 5.8, 6.8, 8.2, 9.2],
                  ["Patient status", "LSTM", "ST-GCN", "LITEMV", "LSTM", "ST-GCN", "LITEMV", "LSTM", "ST-GCN"], fontsize=5.5,
                  rotation=45, ha="right", rotation_mode="anchor")
    ax.legend(handles=[Line2D([], [], marker="D", ls="", color=GRAY, ms=4, label="Patient status alone"),
                       Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="Tested on all participants"),
                       Line2D([], [], marker="o", ls="", color=AQUA, ms=4, label="Tested on patients only"),
                       Line2D([], [], marker="o", ls="", mfc="white", color=AQUA, ms=4, label="Trained and tested on patients only")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=5.8, handletextpad=0.3)
    ax.set_xlim(-0.7, 9.8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("AUC")
    ax.grid(axis="x", visible=False)
    letter(ax, "a")
    # b: KIMORE
    ax = fig.add_subplot(gs[1])
    for i, e in enumerate([1, 2, 3, 4, 5]):
        a_all = max((get("kimore", e, "person", a) for a in ("lstm", "stgcn")), key=lambda t: t[0])
        a_pat = max((get("kimore", e, "person", a, "patients only") for a in ("lstm", "stgcn")), key=lambda t: t[0])
        lab = get("kimore", e, "person", "(patient label only)")
        ax.plot([i - 0.14, i + 0.14], [a_all[0], a_pat[0]], color="#cfcec8", lw=1.2, zorder=1)
        whisker(ax, i - 0.14, a_all, BLUE, "o", 20)
        whisker(ax, i + 0.14, a_pat, AQUA, "o", 20)
        ax.plot([i - 0.34, i + 0.34], [lab[0], lab[0]], color=GRAY, lw=1.6, solid_capstyle="butt", zorder=2)
        # grey band: 95% CI of patient status alone
        ax.add_patch(matplotlib.patches.Rectangle((i - 0.34, lab[1]), 0.68, lab[2] - lab[1], color=GRAY, alpha=0.18, lw=0, zorder=1))
    ax.set_xticks(range(5), ["Arm\nlift", "Lateral\ntilt", "Trunk\nrot.", "Pelvis\nrot.", "Squat"], fontsize=6)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Spearman ρ with clinical score")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="Best deep model, all"),
                       Line2D([], [], marker="o", ls="", color=AQUA, ms=4, label="Best deep model, patients only"),
                       Line2D([], [], color=GRAY, lw=1.6, label="Patient status alone")],
              loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=1, fontsize=6)
    letter(ax, "b", x=-0.16)
    # c: false alarms on correct repetitions, standard and group-balanced training
    ax = fig.add_subplot(gs[2])
    runs = [("irds", "Standard")]
    for exp, tag in (("irds_balanced", "Balanced"), ("irds_reweigh", "Reweighed"), ("irds_gdro", "Group DRO")):
        if os.path.exists(os.path.join(OUT, "deep", exp + ".csv")):
            runs.append((exp, tag))
    fa, labels = {}, []
    for exp, tag in runs:
        d = pd.read_csv(os.path.join(OUT, "deep", exp + ".csv"))
        d = d[(d.protocol == "person") & (d.y == 1)]
        for (a, p), g in d.groupby(["arch", "patient"]):
            fa[(exp, a, p)] = 100 * (g.pred < 0.5).mean()
    x, ticks = 0.0, []
    for arch, an in (("lstm", "LSTM"), ("stgcn", "ST-GCN")):
        for exp, tag in runs:
            for off, p, col in ((-0.19, 0, BLUE), (0.19, 1, ORANGE)):
                v = fa[(exp, arch, p)]
                ax.bar(x + off, v, 0.36, color=col, lw=0, alpha={"irds": 1.0, "irds_balanced": 0.7, "irds_reweigh": 0.5, "irds_gdro": 0.3}[exp])
                ax.text(x + off, v + 0.4, "%.0f" % v, ha="center", va="bottom", fontsize=5.5)
            ticks.append((x, {"Standard": "Std.", "Balanced": "Bal.", "Reweighed": "Rew.", "Group DRO": "DRO"}[tag], an))
            x += 1.0
        x += 0.4
    ax.set_xticks([t for t, _, _ in ticks], [l for _, l, _ in ticks], fontsize=5.5, rotation=45, ha="right", rotation_mode="anchor")
    for an in ("LSTM", "ST-GCN"):
        xs = [t for t, _, a in ticks if a == an]
        ax.text(np.mean(xs), -0.19, an, transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=6.5, color=INK2)
    ax.set_ylabel("Correct reps called incorrect (%)")
    ymax = max(fa.values())
    ax.set_ylim(0, max(22, ymax * 1.2))
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[matplotlib.patches.Patch(color=BLUE, label="Healthy"),
                       matplotlib.patches.Patch(color=ORANGE, label="Patients")], loc="upper left")
    letter(ax, "c", x=-0.30)
    save(fig, "fig2_patient_shortcut")


# ---------------------------------------------------------------- Fig 3
MOVES = ["Abduction", "Extension", "Int./ext.\nrotation", "Scaption"]


def fig3():
    fig = plt.figure(figsize=(W2, 55 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], wspace=0.28)
    ax = fig.add_subplot(gs[0])
    mv = [7, 8, 9, 10]
    combos = [("uiprmd_full", "random", ORANGE, "o", False, -0.24), ("uiprmd_lower", "random", ORANGE, "^", True, -0.08),
              ("uiprmd_full", "person", BLUE, "o", False, 0.08), ("uiprmd_lower", "person", BLUE, "^", True, 0.24)]
    for i, m in enumerate(mv):
        for exp, prot, c, mk, hol, off in combos:
            whisker(ax, i + off, get(exp, m, prot, "stgcn"), c, mk, 20, hollow=hol)
    chance(ax)
    ax.set_xticks(range(4), MOVES)
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("AUC (ST-GCN)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, ms=4, label="Full body, random split"),
                       Line2D([], [], marker="^", ls="", mfc="white", color=ORANGE, ms=4, label="Pelvis + legs only, random split"),
                       Line2D([], [], marker="o", ls="", color=BLUE, ms=4, label="Full body, person-wise"),
                       Line2D([], [], marker="^", ls="", mfc="white", color=BLUE, ms=4, label="Pelvis + legs only, person-wise")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6)
    letter(ax, "a", x=-0.11)
    ax = fig.add_subplot(gs[1])
    pth = os.path.join(OUT, "uiprmd_perm_null.npz")
    if os.path.exists(pth):
        Z = np.load(pth, allow_pickle=False)
        reals = [float(Z["m%02d_real" % m][0]) for m in mv]
        vp = ax.violinplot([Z["m%02d_null" % m] for m in mv], positions=range(4), widths=0.7, showextrema=False)
        for b in vp["bodies"]:
            b.set_facecolor("#d7d6d0")
            b.set_edgecolor("none")
            b.set_alpha(1)
    else:
        P = pd.read_csv(os.path.join(OUT, "control_uiprmd_perm.csv"))
        reals = P.real_AUC.tolist()
        ax.vlines(range(4), P.shuffled_mean, P.shuffled_max, color="#d7d6d0", lw=6)
    rv = os.path.join(OUT, "review_uiprmd.csv")
    if os.path.exists(rv):  # person-level bootstrap CI of the mean within-person AUC
        V = pd.read_csv(rv)
        for i, m in enumerate(mv):
            est = parse(V[V.test.str.startswith("m%02d" % m)].value.iloc[0].replace(", ", "-"))
            whisker(ax, i, est, ORANGE, "o", 22)
    else:
        ax.scatter(range(4), reals, color=ORANGE, s=22, zorder=3, lw=0)
    chance(ax)
    ax.set_xticks(range(4), MOVES)
    ax.set_ylim(0.2, 1.02)
    ax.set_ylabel("Within-person AUC")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker="o", ls="", color=ORANGE, ms=4, label="Observed"),
                       matplotlib.patches.Patch(color="#d7d6d0", label="Label permutation (n = 200)")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2, fontsize=6)
    letter(ax, "b", x=-0.15)
    save(fig, "fig3_uiprmd_fingerprint")


# ---------------------------------------------------------------- Fig 4
def fig4():
    s = pd.read_csv(os.path.join(DATA, "REHAB26-ViewAngles", "Segmentation.csv"), sep=";")
    s = s.sort_values(["exercise_id", "video_id", "person_orientation", "first_frame"])
    sets = [g.correctness_score.eq(1).astype(int).tolist() for _, g in s.groupby(["exercise_id", "video_id", "person_orientation"], sort=False)]
    sets = sorted(sets, key=len)  # sets with 11-12 reps at the bottom, as one block
    L = max(map(len, sets))
    M = np.full((len(sets), L), np.nan)
    for i, v in enumerate(sets):
        M[i, :len(v)] = v
    fig = plt.figure(figsize=(W2, 58 * MM))
    gs = fig.add_gridspec(1, 2, width_ratios=[0.8, 1.2], wspace=0.25)
    ax = fig.add_subplot(gs[0])
    ax.imshow(M, aspect="auto", cmap=ListedColormap([ORANGE, BLUE]), interpolation="nearest", vmin=0, vmax=1)
    ax.set_xticks(range(L), [str(i + 1) for i in range(L)])
    ax.set_xlabel("Repetition within set")
    ax.set_ylabel("Recording set (n = %d)" % len(sets))
    ax.set_yticks([0, len(sets) - 1], ["1", str(len(sets))])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(handles=[matplotlib.patches.Patch(color=BLUE, label="Correct"), matplotlib.patches.Patch(color=ORANGE, label="Incorrect")],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=2)
    letter(ax, "a", x=-0.14)
    ax = fig.add_subplot(gs[1])
    R = pd.read_csv(os.path.join(OUT, "shortcuts_rehab26.csv"))
    ex = [("Ex1", 1, "Arm abduction"), ("Ex6", 6, "Squat"), ("Ex7", 7, "Shoulder extension")]
    series = [("Repetition index only", GRAY, "D", lambda t, e: parse(R[(R.task == t) & R.model.str.startswith("rep number")].AUC.iloc[0]), -0.2),
              ("ST-GCN, single repetition", BLUE, "o", lambda t, e: get("rehab26_single", e, "person", "stgcn"), 0.0),
              ("ST-GCN, with neighboring repetitions", VIOLET, "s", lambda t, e: get("rehab26_context", e, "person", "stgcn"), 0.2)]
    for i, (t, e, _) in enumerate(ex):
        for lab, c, mk, fn, off in series:
            whisker(ax, i + off, fn(t, e), c, mk, 22)
    chance(ax)
    ax.set_xticks(range(3), [n for _, _, n in ex])
    ax.set_ylim(0.45, 1.02)
    ax.set_ylabel("AUC (person-wise)")
    ax.grid(axis="x", visible=False)
    ax.legend(handles=[Line2D([], [], marker=mk, ls="", color=c, ms=4, label=lab) for lab, c, mk, _, _ in series],
              loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=3, fontsize=6)
    letter(ax, "b", x=-0.1)
    save(fig, "fig4_rehab26_order")


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
