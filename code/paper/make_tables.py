"""Paper tables. Reads results/ and writes results/tables/*.csv.

  python make_tables.py
Table 1 is descriptive (checked against the original papers and dataset records, 2026-11-01); Tables 2-4 are filled from result files.
"""
import os
import sys
import re

import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))  # code/
from paths import DATA, OUT  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TAB = os.path.join(OUT, "tables")
os.makedirs(TAB, exist_ok=True)
os.makedirs(TAB, exist_ok=True)


def parse(v):
    m = re.search(r"(-?\d+\.\d+)\s*\[\s*(-?\d+\.\d+)\s*-\s*(-?\d+\.\d+)\s*\]", str(v))
    if m:
        return tuple(float(g) for g in m.groups())
    return (float(re.search(r"-?\d+\.\d+", str(v)).group()), np.nan, np.nan)


def ci(t):
    v, lo, hi = t
    lo, hi = min(lo, v), max(hi, v)
    return f"{v:.2f} [{lo:.2f}, {hi:.2f}]" if np.isfinite(lo) else f"{v:.2f}"


def rng(vals):
    vals = [v[0] if isinstance(v, tuple) else v for v in vals]
    a, b = min(vals), max(vals)
    return f"{a:.2f}" if abs(a - b) < 0.005 else f"{a:.2f}–{b:.2f}"


S = pd.read_csv(os.path.join(OUT, "deep", "summary.csv"))
ST = pd.read_csv(os.path.join(OUT, "strict_results.csv"))


def get(exp, task=None, prot="person", arch="stgcn", subset="all"):
    m = (S.experiment == exp) & (S.protocol == prot) & (S.arch == arch) & (S.subset == subset)
    if task is not None:
        m &= S.task.astype(str) == str(task)
    return parse(S[m].value.iloc[0])


def strict(ds, task, variant, col="AUC"):
    r = ST[(ST.dataset == ds) & (ST.task == task) & (ST.variant == variant)]
    return parse(r[col].iloc[0])


def csvrow(f, task, model, col="AUC"):
    d = pd.read_csv(os.path.join(OUT, f))
    return parse(d[(d.task == task) & (d.model == model)][col].iloc[0])


# ---------------------------------------------------------------- Table 1
T1 = pd.DataFrame([
    ["REHAB24-6", "2024", "10 (0/10)ᵃ", "2 RGB cameras + optical mocap", "Arm abduction, arm V-W (RGB); 6 exercises (mocap)", "1,072 reps in total; we used 386 arm reps (RGB) and 1,057 reps with valid mocap", "Binary", "Deliberate", "Correct block, then incorrect block (in annotations; not stated in paper)"],
    ["REHAB26-ViewAngles", "2026", "5 (volunteers)", "3 RGB cameras (MediaPipe pose)", "Arm abduction, squat, shoulder extension", "1,665 reps (1,650 in paper); one camera view per repetition", "Binary + error type", "Deliberate", "Interleaved, but 152 of the 165 sets (all with 10 reps) follow the same order: 3 correct, 5 incorrect, 2 correct"],
    ["UI-PRMD", "2018", "10 (0/10)", "Vicon (39 markers) + Kinect", "4 shoulder movements (of 10)", "800 episodes (Vicon)", "Binary", "Deliberately non-optimal", "Correct and incorrect released as separate sets; session order not reported"],
    ["IntelliRehabDS", "2021", "29 (15/14)ᵇ", "Kinect One (v2)", "9 upper/lower-limb gestures", "2,589 files (2,575 used: 12 unusable and 2 shorter than 16 frames removed)", "Binary; 12 marked unusable", "Natural", "98% of incorrect sequences from patients"],
    ["KIMORE", "2019", "78 (34/44)ᶜ", "Kinect v2", "5 low-back exercises", "1 recording (5 reps) / person / exercise", "Clinical total score 10–50", "Natural", "Patients score lower, experts highest (paper Fig. 4); healthy median 42–46 vs patients 25–35"],
    ["KERAAL†", "2024", "21 (12/9)ᵈ", "Kinect v2 + RGB (+ Vicon, group 3)", "3 low-back exercises", "812 annotated recordings", "Binary + error type", "Natural (patients) + simulated (3 healthy)", "Simulating group 25% correct for every person"],
    ["UCDHE†", "2023", "52 (0/52)ᵉ", "RGB video, 2-D pose (OpenPose, 8 joints)", "Military press, rowing", "4,289 reps", "Binary (merged from 4–5 classes)", "Deliberate", "No patients; repetition order not released (control for the patient shortcut)"],
], columns=["Dataset", "Year", "People (patients/healthy)", "Sensor", "Exercises used", "Units", "Label", "Errors", "Recording design relevant to shortcuts"])

# ---------------------------------------------------------------- Table 2
irds_fa = pd.read_csv(os.path.join(OUT, "deep", "irds.csv"))
irds_fa = irds_fa[(irds_fa.protocol == "person") & (irds_fa.y == 1)]
fa = {(a, p): 100 * (g.pred < 0.5).mean() for (a, p), g in irds_fa.groupby(["arch", "patient"])}
best = lambda exp, t, prot="person", sub="all": max((get(exp, t, prot, a, sub) for a in ("lstm", "stgcn")), key=lambda x: x[0])
perm = pd.read_csv(os.path.join(OUT, "control_uiprmd_perm.csv"))
k_lab = [get("kimore", e, "person", "(patient label only)") for e in range(1, 6)]
k_all = [best("kimore", e) for e in range(1, 6)]
k_pat = [best("kimore", e, "person", "patients only") for e in range(1, 6)]
rv = os.path.join(OUT, "review_kimore.csv")
k_part = []
if os.path.exists(rv):  # partial rho controlling for patient status, better of the two models per exercise
    K = pd.read_csv(rv)
    for e in range(1, 6):
        vals = [parse(v.replace(", ", "-")) for v in K[K.test.str.startswith("Es%d " % e)].value]
        k_part.append(max(vals, key=lambda x: x[0]))
ri = os.path.join(OUT, "review_irds.csv")
irds_gain = ""
if os.path.exists(ri):
    I = pd.read_csv(ri)
    g = [parse(v.replace(", ", "-"))[0] for v in I[I.test == "AUC gain of adding the score"].value]
    irds_gain = "; adding the model score to patient status: +%.2f" % max(g)
T2 = pd.DataFrame([
    ["REHAB24-6 (RGB)", "Repetition position in recording", rng([strict("REHAB24-6", "Ex1", "rep position ONLY"), strict("REHAB24-6", "Ex2", "rep position ONLY")]),
     rng([best("rehab24_cam", 1), best("rehab24_cam", 2)]),
     "Position + checks (adds the probe) " + rng([strict("REHAB24-6", "Ex1", "rep position + checks"), strict("REHAB24-6", "Ex2", "rep position + checks")]),
     rng([best("rehab24_cam", t, "random") for t in (1, 2)])],
    ["REHAB26-ViewAngles", "Repetition number (one-hot)", rng([csvrow("shortcuts_rehab26.csv", t, "rep number ONLY (label design)") for t in ("Ex1", "Ex6", "Ex7")]),
     rng([best("rehab26_single", e) for e in (1, 6, 7)]),
     "With neighboring repetitions (adds order) " + rng([best("rehab26_context", e) for e in (1, 6, 7)]),
     rng([best("rehab26_random", e, "random") for e in (1, 6, 7)])],
    ["UI-PRMD", "Pelvis + legs only (random split); rigid marker spacing (within person)",
     rng([best("uiprmd_lower", m, "random") for m in (7, 8, 9, 10)]) + f"; {perm.real_AUC.min():.2f}–{perm.real_AUC.max():.2f}",
     rng([best("uiprmd_full", m) for m in (7, 8, 9, 10)]),
     "Pelvis + legs only, person-wise " + rng([best("uiprmd_lower", m) for m in (7, 8, 9, 10)]) +
     f"; permutation null max {perm.shuffled_max.max():.2f}",
     rng([best("uiprmd_full", m, "random") for m in (7, 8, 9, 10)])],
    ["IntelliRehabDS", "Patient status", ci(get("irds", None, "person", "(patient label only)")),
     ci(best("irds", None)),
     "Patients only " + ci(best("irds", None, "person", "patients only")) + irds_gain +
     "; LITEMV (Rehab-Pile) " + "%.2f" % get("irds_litemv", None, "person", "litemv")[0] +
     f"; correct reps called incorrect: patients {fa[('lstm', 1)]:.0f}–{fa[('stgcn', 1)]:.0f}%, healthy {fa[('lstm', 0)]:.0f}–{fa[('stgcn', 0)]:.0f}%",
     ci(best("irds", None, "random"))],
    ["KIMORE (ρ)", "Patient status", rng(k_lab), rng(k_all),
     "Patients only " + rng(k_pat) + ("; partial ρ given patient status " + rng(k_part) if k_part else "") +
     "; LITEMV (Rehab-Pile) " + rng([get("kimore_litemv", e, "person", "litemv") for e in range(1, 6)]), "n/a"],
    ["KERAAL", "Recording group", rng([csvrow("shortcuts_keraal.csv", t, "group label only") for t in ("CTK", "ELK", "RTK")]) if os.path.exists(os.path.join(OUT, "shortcuts_keraal.csv")) else "0.54–0.66",
     rng([best("keraal", t) for t in ("CTK", "ELK", "RTK")]),
     "Simulating group only " + rng([best("keraal", t, "person", "simulating group only (P)") for t in ("CTK", "ELK", "RTK")]) +
     "; patients only " + rng([best("keraal", t, "person", "patients only (R)") for t in ("CTK", "ELK", "RTK")]),
     rng([best("keraal", t, "random") for t in ("CTK", "ELK", "RTK")])],
    ["UCDHE (control)", "No patient group; repetition order not released", "—", rng([best("ucdhe", t) for t in ("MP", "Rowing")]), "—",
     rng([best("ucdhe", t, "random") for t in ("MP", "Rowing")])],
], columns=["Dataset", "Movement-free probe", "Probe alone", "Deep model, person-wise", "Further tests", "Deep model, random split"])

# ---------------------------------------------------------------- Table 3
T3 = pd.DataFrame([
    ["REHAB24-6 (RGB)", "Arm abduction", ci(strict("REHAB24-6", "Ex1", "without duration")), ci(get("rehab24_cam", 1, arch="lstm")), ci(get("rehab24_cam", 1))],
    ["REHAB24-6 (RGB)", "Arm V-W", ci(strict("REHAB24-6", "Ex2", "without duration")), ci(get("rehab24_cam", 2, arch="lstm")), ci(get("rehab24_cam", 2))],
] + [["UI-PRMD", n, ci(strict("UI-PRMD", f"m{m:02d} {full}", "without duration")), ci(get("uiprmd_full", m, arch="lstm")), ci(get("uiprmd_full", m))]
     for m, n, full in ((7, "Abduction", "shoulder abduction"), (8, "Extension", "shoulder extension"),
                        (9, "Int./ext. rotation", "shoulder int/ext rotation"), (10, "Scaption", "shoulder scaption"))] + [
    ["IntelliRehabDS", "9 gestures (generic features)", ci(csvrow("shortcuts_irds.csv", "9 gestures pooled", "movement features (all people)")),
     ci(get("irds", None, arch="lstm")), ci(get("irds", None))],
] + [["KIMORE (ρ)", n, ci(strict("KIMORE", f"Es{e}", "without duration and rep count", "rho_all")), ci(get("kimore", e, arch="lstm")), ci(get("kimore", e))]
     for e, n in ((1, "Arm lift"), (2, "Lateral tilt"), (3, "Trunk rotation"), (4, "Pelvis rotation"), (5, "Squat"))],
    columns=["Dataset", "Exercise", "Interpretable checks", "LSTM", "ST-GCN"])

# ---------------------------------------------------------------- Table 4
T4 = pd.DataFrame([
    ["Population (patient status)", "IntelliRehabDS (KERAAL: weak)", "Strong: probe 0.84, close to the models; correct reps called incorrect 14–18% (patients) vs 1–3% (healthy)",
     "Report a patient-status-only baseline, results on patients only, results conditioned on patient status, and false alarms on correct reps by group",
     "Record correct and incorrect reps from patients; record healthy errors"],
    ["Group-linked clinical score", "KIMORE", "Patients really score lower; patient status alone (ρ 0.55–0.66) matches the models in 3 of 5 exercises",
     "Report the partial correlation given patient status and results on patients only", "Record the full score range inside each group; release the group"],
    ["Recording session", "UI-PRMD", "Strong inside each person (0.92–0.99); weak across people (0.61–0.69)", "Person-wise splits only; probe with movement-uninvolved body parts; permutation control",
     "Record correct and incorrect reps in the same session without re-applying markers"],
    ["Label order (block design)", "REHAB26, REHAB24-6", "REHAB26: probe 0.98–0.99, used a little by context models (+0.04–0.12); REHAB24-6: design risk only",
     "Report a rep-index-only baseline; avoid context windows that cross repetitions", "Randomize the order of correct and incorrect repetitions"],
    ["Simulated vs natural errors", "KERAAL, (UI-PRMD, REHAB24-6/26, UCDHE)", "Moderate", "Report performance on natural (patient) errors separately",
     "Include natural errors from the target population"],
    ["Subject leakage (random split)", "All multi-rep datasets incl. UCDHE", "+0.05 to +0.32 AUC", "Person-wise (grouped) cross-validation with person-level CIs",
     "Release person IDs, group, order and session metadata"],
    ["Repetition duration", "IntelliRehabDS (REHAB24-6: not consistent)", "Moderate in IntelliRehabDS (0.70); REHAB24-6 0.35 and 0.57, direction differs between people", "Report a duration-only baseline; exclude or justify duration features", "—"],
], columns=["Shortcut", "Datasets", "Strength under person-wise evaluation", "Evaluation recommendation", "Data-collection recommendation"])

TABLES = [("table1_datasets", "Table 1. Datasets and recording designs relevant to shortcuts (six audited datasets and one control). Facts checked against the original papers and dataset records. "
           "ᵃHealthy (some with past injuries, none acute); 9 people have arm-exercise data. "
           "ᵇThe release also contains patient 217, excluded by the authors (recording stopped); our analysis keeps it (61 sequences; removing it changes AUCs by ≤0.01). "
           "ᶜ12 expert + 32 non-expert healthy; 10 stroke, 16 Parkinson, 8 back pain; 77 have usable data. "
           "ᵈWe use the annotated subset: 6 patients, 6 healthy, 3 healthy simulating errors. "
           "ᵉSingh et al. 2023 (military press, DAMI; rowing, ECML-PKDD) report 53/54 volunteers; 52 is the Rehab-Pile count. "
           "†As distributed in the Rehab-Pile archive. ", T1),
          ("table2_shortcuts", "Table 2. Movement-free probes compared with deep models (AUC unless ρ). Deep-model columns give the better of LSTM and ST-GCN for each exercise; ranges are over exercises; brackets give 95% person-level bootstrap CIs.", T2),
          ("table3_interpretable_vs_deep", "Table 3. Interpretable kinematic checks versus deep models under person-wise evaluation (AUC or Spearman ρ, 95% CI).", T3),
          ("table4_checklist", "Table 4. Shortcut audit checklist.", T4)]


if __name__ == "__main__":
    for name, cap, df in TABLES:
        df.to_csv(os.path.join(TAB, name + ".csv"), index=False)
        print(cap)
        print(df.to_string(index=False), "\n")
    print("written to", TAB)
