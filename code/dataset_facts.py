"""Dataset facts quoted in the paper (Section III), recomputed from the released files.

- REHAB24-6: number of recordings that start with a block of correct repetitions before the first incorrect one;
  people and repetitions.
- REHAB26-ViewAngles: number of sets, set sizes and how many sets follow the order 3 correct, 5 incorrect, 2 correct.
- IntelliRehabDS: files, label counts, sequences used, patients and share of incorrect sequences from patients.
- KERAAL (Rehab-Pile): share of correct recordings per group.

Output: results/dataset_facts.txt (also printed).
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

from paths import DATA, OUT


def rehab24():
    s = pd.read_csv(os.path.join(DATA, "REHAB24-6", "Segmentation.csv"), sep=";")
    both, block_first = 0, 0
    for _, g in s.groupby("video_id"):
        c = g.sort_values("repetition_number").correctness.tolist()
        if 0 in c and 1 in c:
            both += 1
            first_bad = c.index(0)
            block_first += int(c[0] == 1 and all(c[:first_bad]))
    arm = s[s.exercise_id.isin([1, 2])]
    return ["REHAB24-6: %d people, %d repetitions, %d with valid mocap" % (s.person_id.nunique(), len(s), (s.mocap_erroneous == 0).sum()),
            "REHAB24-6: arm exercises %d repetitions, %d people" % (len(arm), arm.person_id.nunique()),
            "REHAB24-6: %d of %d recordings with both classes start with a block of correct repetitions" % (block_first, both)]


def rehab26():
    s = pd.read_csv(os.path.join(DATA, "REHAB26-ViewAngles", "Segmentation.csv"), sep=";")
    s = s.sort_values(["video_id", "person_orientation", "first_frame"])
    pat = [tuple(int(v == 1) for v in g.correctness_score) for _, g in s.groupby(["video_id", "person_orientation"])]
    target = (1, 1, 1, 0, 0, 0, 0, 0, 1, 1)
    sizes = pd.Series([len(p) for p in pat]).value_counts().sort_index().to_dict()
    return ["REHAB26: %d repetitions in %d sets; set sizes %s" % (len(s), len(pat), sizes),
            "REHAB26: %d sets follow 3 correct, 5 incorrect, 2 correct" % sum(p == target for p in pat)]


def irds():
    files = glob.glob(os.path.join(DATA, "IntelliRehabDS", "extracted", "SkeletonData", "Simplified", "*.txt"))
    rows = []
    for p in files:
        subj, _, gest, rep, lab, pos = os.path.basename(p)[:-4].split("_")
        n = len(np.loadtxt(p, delimiter=",", ndmin=2))
        rows.append(dict(subject=subj, label=lab, frames=n, patient=subj.startswith("2"), position=pos.lower()))
    d = pd.DataFrame(rows)
    used = d[d.label.isin(["1", "2"]) & (d.frames >= 16)]
    inc = used[used.label == "2"]
    return ["IRDS: %d files; labels %s" % (len(d), d.label.value_counts().sort_index().to_dict()),
            "IRDS: %d sequences used (labels 1/2, >= 16 frames); %d people, %d patients (incl. 217)"
            % (len(used), used.subject.nunique(), used[used.patient].subject.nunique()),
            "IRDS: %.1f%% of incorrect sequences from patients; patient 217 sequences: %d"
            % (100 * inc.patient.mean(), (used.subject == "217").sum()),
            "IRDS: positions %s" % used.position.value_counts().to_dict()]


def keraal():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "deep"))
    import data as D
    items = D.rehabpile("KERAAL_clf_bn", ["CTK", "ELK", "RTK"], "C")
    d = pd.DataFrame([dict(task=it["task"], group=it["group"], y=it["y"], person=it["person"]) for it in items])
    out = ["KERAAL: %d recordings; people per group %s" % (len(d), d.groupby("group").person.nunique().to_dict())]
    share = d.groupby(["group", "task"]).y.mean().unstack().round(2)
    out.append("KERAAL: share correct by group and exercise\n" + share.to_string())
    return out


if __name__ == "__main__":
    lines = []
    for f in (rehab24, rehab26, irds, keraal):
        lines += f()
    txt = "\n".join(lines)
    print(txt)
    open(os.path.join(OUT, "dataset_facts.txt"), "w", encoding="utf-8").write(txt + "\n")
