# Hidden shortcuts in rehabilitation exercise assessment datasets

Code to reproduce the results of *"Right for the Wrong Reasons? Hidden Shortcuts in Rehabilitation Exercise
Assessment Datasets"* (submitted to IEEE Journal of Biomedical and Health Informatics, 2026).

## Setup

Python 3.12. Install the packages:

```
pip install -r requirements.txt
```

The datasets are not included. Download them from the sources cited in the paper (REHAB24-6, REHAB26-ViewAngles,
UI-PRMD, IntelliRehabDS, KIMORE, and the Rehab-Pile archive for KERAAL and UCDHE). Put them in `./data`, or set
`REHAB_DATA`, with this layout:

```
data/
  REHAB24-6/            Segmentation.csv, extracted/Ex1..Ex6/
  REHAB26-ViewAngles/   Segmentation.csv, extracted/mediapipe/
  UI-PRMD/extracted/
  IntelliRehabDS/extracted/SkeletonData/Simplified/
  KIMORE/extracted/KiMoRe/
  RehabPile/KERAAL_clf_bn/, RehabPile/UCDHE_clf_bn/
```

## Run

```
python run_all.py               # everything; the deep models take about 3 hours on one GPU
python run_all.py --skip-deep   # everything except training the deep models (steps that need their predictions are skipped)
```

Results are written to `./results` (or `REHAB_OUT`): probe and control results as CSV files, out-of-fold
predictions in `results/deep/`, and the figures and tables of the paper in `results/figures/` and `results/tables/`.

Deep-model results can differ slightly between runs and GPUs. All other steps give the same numbers every time.
