"""Three checks on the conclusions, all with the reference model (random forest, shape features).

1. Label-shuffling null for the held-out-tests protocol.
2. Leave-one-test-out with and without same-flow tests in the training set.
3. Held-out layout after standardising features within each layout.

Run from the repository root:  python scripts/run_checks.py   (about 5 minutes)
"""
import sys
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev  # noqa: E402
import features as ft  # noqa: E402
import leakdata as ld  # noqa: E402

RESULTS = ld.REPO_ROOT / "results"
N_PERMUTATIONS = 200
SHAPE = ft.SHAPE_FEATURES


def load(stype):
    feats = pd.read_parquet(ld.PROCESSED / "features.parquet")
    return feats[(feats.sensor_type == stype) & (feats.leak != "BG")].reset_index(drop=True)


def rf_proba(x_train, y_train, x_test, n_classes):
    clf = ev.make_model("rf").fit(x_train, y_train)
    proba = np.zeros((len(x_test), n_classes))
    proba[:, clf.classes_] = clf.predict_proba(x_test)
    return proba


def recording_score(files, proba, y):
    table = pd.DataFrame(proba)
    table["file"], table["y"] = np.asarray(files), y
    rec = table.groupby("file").mean()
    return balanced_accuracy_score(rec.y.round().astype(int), rec.drop(columns="y").to_numpy().argmax(axis=1))


def permutation_run(stype, task, seed):
    """Held-out-tests score with class labels shuffled between tests of the same layout.

    seed < 0 keeps the true labels. Folds are the ones used in the real run.
    """
    warnings.filterwarnings("ignore")
    d = load(stype)
    conditions = np.sort(d.condition.unique())
    relabel = {}
    rng = np.random.default_rng(max(seed, 0))
    for topo in ld.TOPOLOGIES:
        group = [c for c in conditions if c.startswith(topo)]
        true = [c.split("|")[1] for c in group]
        relabel.update(zip(group, true if seed < 0 else rng.permutation(true)))
    y = ev.target(d.assign(leak=d.condition.map(relabel)), task)
    x = d[SHAPE].to_numpy(np.float32)
    n_classes = len(ev.TASKS[task])
    folds = ev.fold_ids(d, "grouped_condition")
    proba = np.zeros((len(d), n_classes))
    for fold in np.unique(folds):
        test = folds == fold
        proba[test] = rf_proba(x[~test], y[~test], x[test], n_classes)
    return recording_score(d.file, proba, y)


def leave_one_test_out(stype, condition, variant):
    """Predict one physical test from the others, under three training sets."""
    warnings.filterwarnings("ignore")
    d = load(stype)
    x, y = d[SHAPE].to_numpy(np.float32), ev.target(d, "type")
    topo, leak, flow = condition.split("|")
    test = (d.condition == condition).to_numpy()
    train = ~test
    if variant == "without same-flow tests":
        train &= (d.flow != flow).to_numpy()
    elif variant == "without 9 random other tests":
        # Size control: drop as many tests as the variant above, but from other
        # flow conditions, and never the test's own class in its own layout.
        rng = np.random.default_rng(zlib.crc32(condition.encode()))
        candidates = [
            c for c in sorted(d.condition.unique())
            if c.split("|")[2] != flow and not c.startswith(f"{topo}|{leak}|")
        ]
        train &= ~d.condition.isin(rng.choice(candidates, 9, replace=False)).to_numpy()
    proba = pd.DataFrame(rf_proba(x[train], y[train], x[test], 5))
    proba["file"] = d.file.to_numpy()[test]
    rec = proba.groupby("file").mean()
    return [(stype, variant, condition, f, int(p.argmax()), ld.LEAKS.index(leak)) for f, p in zip(rec.index, rec.to_numpy())]


def held_out_layout(stype, task, mode):
    warnings.filterwarnings("ignore")
    d = load(stype)
    x = d[SHAPE].astype(float)
    if mode != "raw features":
        group = d.topology if mode == "standardised per layout" else d.sensor + d.topology
        x = (x - x.groupby(group).transform("mean")) / (x.groupby(group).transform("std") + 1e-9)
    x, y = x.to_numpy(np.float32), ev.target(d, task)
    n_classes = len(ev.TASKS[task])
    proba = np.zeros((len(d), n_classes))
    for topo in ld.TOPOLOGIES:
        test = (d.topology == topo).to_numpy()
        proba[test] = rf_proba(x[~test], y[~test], x[test], n_classes)
    return recording_score(d.file, proba, y)


def main():
    pool = Parallel(n_jobs=12, verbose=1)

    jobs = [(s, t, seed) for s in "APH" for t in ev.TASKS for seed in range(-1, N_PERMUTATIONS)]
    scores = pool(delayed(permutation_run)(*j) for j in jobs)
    perm = pd.DataFrame(jobs, columns=["sensor_type", "task", "seed"]).assign(score=scores)
    perm.to_csv(RESULTS / "check_permutation.csv", index=False, float_format="%.4f")

    conditions = sorted(load("A").condition.unique())
    variants = ["all other tests", "without same-flow tests", "without 9 random other tests"]
    jobs = [(s, c, v) for s in "APH" for c in conditions for v in variants]
    parts = pool(delayed(leave_one_test_out)(*j) for j in jobs)
    loto = pd.DataFrame([r for p in parts for r in p], columns=["sensor_type", "variant", "condition", "file", "pred", "y"])
    loto.to_csv(RESULTS / "check_leave_one_test_out.csv", index=False)

    modes = ["raw features", "standardised per layout", "standardised per sensor and layout"]
    jobs = [(s, t, m) for s in "APH" for t in ev.TASKS for m in modes]
    scores = pool(delayed(held_out_layout)(*j) for j in jobs)
    layout = pd.DataFrame(jobs, columns=["sensor_type", "task", "mode"]).assign(score=scores)
    layout.to_csv(RESULTS / "check_layout_standardisation.csv", index=False, float_format="%.4f")
    print(layout.pivot_table(index=["task", "sensor_type"], columns="mode", values="score").round(2))


if __name__ == "__main__":
    main()
