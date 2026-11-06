"""Run the whole analysis in order and rebuild the paper's figures and tables.

  python run_all.py                  # everything (deep models take about 3 hours on one RTX 5090)
  python run_all.py --skip-deep      # everything except training the deep models
  python run_all.py --quick-deep     # deep models with 1 epoch and 1 seed (pipeline test only, numbers not valid)
  python run_all.py --only figures   # one step (see STEPS below)

Datasets are read from REHAB_DATA (default ./data), results go to REHAB_OUT (default ./results).
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.join(HERE, "code")
DEEP = ["rehab24_cam", "rehab24_mocap", "rehab26_single", "rehab26_context", "rehab26_random", "irds", "irds_patients",
        "irds_balanced", "irds_reweigh", "irds_gdro", "uiprmd_full", "uiprmd_lower", "kimore", "keraal", "keraal_patients", "ucdhe"]

STEPS = [
    ("facts", ["dataset_facts.py"]),
    ("mediapipe", ["run_mediapipe.py"]),                 # REHAB24-6 RGB videos -> MediaPipe pose
    ("rehab24", ["evaluate2.py"]),                        # REHAB24-6 interpretable features
    ("uiprmd", ["eval_uiprmd.py"]),
    ("kimore", ["eval_kimore.py"]),
    ("strict", ["strict_all.py"]),                        # interpretable checks with strict LOPO + probes
    ("probes_irds", ["shortcuts.py", "irds"]),
    ("probes_uiprmd_kimore", ["shortcuts.py", "uiprmd", "kimore"]),
    ("probes_rehab26", ["rehab26_shortcuts.py"]),
    ("probes_keraal", ["keraal_probe.py"]),
    ("controls", ["controls.py", "uiprmd_perm", "kimore_take", "context"]),
    ("deep", None),
    ("litemv", ["deep/run_litemv.py", "irds", "kimore"]),   # needs TensorFlow + the Rehab-Pile code (see README)
    ("summary", ["deep/summarize.py"]),
    ("conditional", ["conditional_checks.py"]),
    ("figures", ["paper/make_figures.py"]),
    ("tables", ["paper/make_tables.py"]),
]


def run(args):
    cmd = [sys.executable, os.path.join(CODE, args[0])] + args[1:]
    t0 = time.time()
    print(">>", " ".join(args), flush=True)
    subprocess.run(cmd, cwd=CODE, check=True, env=dict(os.environ, PYTHONIOENCODING="utf-8"))
    print("   done in %.0f s" % (time.time() - t0), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-deep", action="store_true")
    ap.add_argument("--quick-deep", action="store_true")
    ap.add_argument("--skip-litemv", action="store_true", help="skip the Rehab-Pile LITEMV baseline")
    ap.add_argument("--skip-mediapipe", action="store_true", help="reuse results/mediapipe")
    ap.add_argument("--only", nargs="*", help="run only these steps")
    a = ap.parse_args()
    out = os.environ.get("REHAB_OUT", os.path.join(HERE, "results"))
    for name, args in STEPS:
        if name in ("summary", "conditional", "figures", "tables") and not glob.glob(os.path.join(out, "deep", "*.csv")):
            print("-- skipping %s: no deep-model predictions in %s" % (name, os.path.join(out, "deep")), flush=True)
            continue
        if a.only and name not in a.only:
            continue
        if name == "mediapipe" and a.skip_mediapipe:
            continue
        if name == "litemv" and (a.skip_deep or a.skip_litemv):
            continue
        if name == "deep":
            if a.skip_deep:
                continue
            extra = ["--epochs", "1", "--seeds", "1"] if a.quick_deep else []
            for e in DEEP:
                run(["deep/run.py", e] + extra)
            continue
        run(args)


if __name__ == "__main__":
    main()
