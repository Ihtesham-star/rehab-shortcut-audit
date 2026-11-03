"""Where the datasets are and where results are written.

Set REHAB_DATA to the folder that holds the datasets (layout in README.md) and, optionally,
REHAB_OUT to the folder for results. Defaults: <repo>/data and <repo>/results.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("REHAB_DATA", os.path.join(ROOT, "data"))
OUT = os.environ.get("REHAB_OUT", os.path.join(ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
