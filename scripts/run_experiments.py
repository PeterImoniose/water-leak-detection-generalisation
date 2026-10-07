"""Run every model under every evaluation protocol and save out-of-fold predictions.

Run from the repository root:  python scripts/run_experiments.py
Needs data/processed/features.parquet and spectrograms.npz (see build_features.py).
"""
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import evaluate as ev  # noqa: E402
import leakdata as ld  # noqa: E402

RESULTS = ld.REPO_ROOT / "results"
CLASSICAL = ["logreg", "rf", "lgbm"]
META = ["file", "sensor_type", "sensor", "topology", "leak", "flow", "noise", "condition", "window"]


def run_fold(df, x, task, model, feature_set, protocol, fold, test_mask):
    warnings.filterwarnings("ignore")
    y = ev.target(df, task)
    train, test = ~test_mask, test_mask
    n_classes = len(ev.TASKS[task])
    if model == "cnn":
        proba = ev.fit_predict_cnn(x[train], y[train], x[test], n_classes)
    else:
        clf = ev.make_model(model).fit(x[train], y[train])
        proba = np.zeros((test.sum(), n_classes))
        proba[:, clf.classes_] = clf.predict_proba(x[test])
    out = df.loc[test, META].copy()
    out["y"] = y[test]
    for i, col in enumerate(ev.proba_columns(task)):
        out[col] = proba[:, i].astype(np.float32)
    out["task"], out["model"], out["feature_set"], out["protocol"], out["fold"] = task, model, feature_set, protocol, fold
    return out


def main():
    feats = pd.read_parquet(ld.PROCESSED / "features.parquet")
    specs = np.load(ld.PROCESSED / "spectrograms.npz")
    jobs = []
    for stype in "APH":
        rows = (feats.sensor_type == stype).to_numpy()
        keep = (feats.leak[rows] != "BG").to_numpy()  # drop the hydrophone background-noise files
        df = feats[rows][keep].reset_index(drop=True)
        inputs = {name: df[cols].to_numpy(np.float32) for name, cols in ev.FEATURE_SETS.items()}
        inputs["spectrogram"] = specs[stype][keep].astype(np.float32)
        for protocol in ev.PROTOCOLS:
            ids = ev.fold_ids(df, protocol)
            for fold in sorted(set(ids)):
                mask = ids == fold
                for task in ev.TASKS:
                    for fs_name in ev.FEATURE_SETS:
                        for model in CLASSICAL:
                            jobs.append((df, inputs[fs_name], task, model, fs_name, protocol, fold, mask))
                    jobs.append((df, inputs["spectrogram"], task, "cnn", "spectrogram", protocol, fold, mask))
    print(f"{len(jobs)} fits")
    t0 = time.time()
    # CNN fits are the slow ones; run them first so the pool stays busy.
    jobs.sort(key=lambda j: j[3] != "cnn")
    parts = Parallel(n_jobs=7, verbose=1)(delayed(run_fold)(*j) for j in jobs)
    print(f"{time.time() - t0:.0f} s")

    RESULTS.mkdir(exist_ok=True)
    oof = pd.concat(parts, ignore_index=True)
    for col in ["task", "model", "feature_set", "protocol", "fold", "sensor_type", "sensor", "topology", "leak", "flow", "noise"]:
        oof[col] = oof[col].astype("category")
    oof.to_parquet(RESULTS / "oof_predictions.parquet", index=False)

    keys = ["sensor_type", "task", "model", "feature_set", "protocol"]
    overall = [dict(zip(keys, k), **ev.score(g, k[1])) for k, g in oof.groupby(keys, observed=True)]
    pd.DataFrame(overall).to_csv(RESULTS / "metrics.csv", index=False, float_format="%.4f")
    per_fold = [dict(zip(keys + ["fold"], k), **ev.score(g, k[1])) for k, g in oof.groupby(keys + ["fold"], observed=True)]
    pd.DataFrame(per_fold).to_csv(RESULTS / "metrics_by_fold.csv", index=False, float_format="%.4f")
    print(pd.DataFrame(overall).to_string())


if __name__ == "__main__":
    main()
