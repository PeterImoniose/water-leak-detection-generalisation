"""Evaluation protocols, models and metrics.

Every hyperparameter here is fixed in advance. Nothing is tuned on held-out
data, so the reported scores are not selected from a search.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import features as ft
from leakdata import LEAKS

SEED = 0
PROTOCOLS = ["random_window", "grouped_condition", "leave_flow_out", "leave_topology_out"]
PROTOCOL_NAMES = {
    "random_window": "Random windows (leaky)",
    "grouped_condition": "Held-out tests",
    "leave_flow_out": "Held-out flow condition",
    "leave_topology_out": "Held-out pipe layout",
}
TASKS = {"detect": ["no-leak", "leak"], "type": LEAKS}
FEATURE_SETS = {
    "shape": ft.SHAPE_FEATURES,
    "shape+amplitude": ft.SHAPE_FEATURES + ft.AMPLITUDE_FEATURES,
}


def target(df, task):
    if task == "detect":
        return (df.leak != "NL").astype(int).to_numpy()
    return df.leak.map({k: i for i, k in enumerate(LEAKS)}).to_numpy()


def fold_ids(df, protocol):
    """Test-fold label for every row under the given protocol."""
    if protocol == "random_window":
        ids = np.empty(len(df), dtype=object)
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for k, (_, test) in enumerate(skf.split(df, df.leak)):
            ids[test] = f"fold{k}"
        return ids
    if protocol == "grouped_condition":
        # Folds are assigned per physical test (topology, leak, flow), stratified
        # by leak class. The assignment depends only on the condition names, so it
        # is identical for every sensor type and predictions can be fused later.
        cond = pd.DataFrame({"condition": sorted(df.condition.unique())})
        cond["leak"] = cond.condition.str.split("|").str[1]
        cond["fold"] = ""
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        for k, (_, test) in enumerate(skf.split(cond, cond.leak)):
            cond.loc[test, "fold"] = f"fold{k}"
        return df.condition.map(cond.set_index("condition").fold).to_numpy()
    if protocol == "leave_flow_out":
        return df.flow.to_numpy()
    if protocol == "leave_topology_out":
        return df.topology.to_numpy()
    raise ValueError(protocol)


def make_model(name):
    if name == "logreg":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=3000, class_weight="balanced")
        )
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample", random_state=SEED, n_jobs=1
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=SEED,
            n_jobs=1,
            verbose=-1,
        )
    raise ValueError(name)


def fit_predict_cnn(x_train, y_train, x_test, n_classes, seed=SEED, epochs=15):
    """Small 2D CNN on log-spectrograms. Returns class probabilities for x_test."""
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(2)

    def block(c_in, c_out):
        return [nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU()]

    net = nn.Sequential(
        *block(1, 16),
        nn.MaxPool2d(2),
        *block(16, 32),
        nn.MaxPool2d(2),
        *block(32, 64),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(64, n_classes),
    )
    xt = torch.from_numpy(x_train.astype(np.float32))[:, None]
    yt = torch.from_numpy(y_train.astype(np.int64))
    counts = np.bincount(y_train, minlength=n_classes)
    weight = torch.tensor(len(y_train) / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * int(np.ceil(len(xt) / 64)))
    gen = torch.Generator().manual_seed(seed)
    net.train()
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), 64):
            b = perm[i : i + 64]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss_fn(net(xt[b]), yt[b]).backward()
            opt.step()
            sched.step()
    net.eval()
    with torch.no_grad():
        logits = net(torch.from_numpy(x_test.astype(np.float32))[:, None])
        return torch.softmax(logits, dim=1).numpy()


def proba_columns(task):
    return [f"p_{c}" for c in TASKS[task]]


def score(oof, task):
    """Metrics for one out-of-fold prediction table (one row per window)."""
    cols = proba_columns(task)
    y = oof.y.to_numpy()
    p = oof[cols].to_numpy()
    pred = p.argmax(axis=1)
    out = {
        "n_windows": len(oof),
        "window_bal_acc": balanced_accuracy_score(y, pred),
        "window_macro_f1": f1_score(y, pred, average="macro"),
    }
    # Recording level: average the window probabilities of each file.
    rec = oof.groupby("file").agg({**{c: "mean" for c in cols}, "y": "first"})
    out["n_recordings"] = len(rec)
    out["recording_bal_acc"] = balanced_accuracy_score(rec.y, rec[cols].to_numpy().argmax(axis=1))
    if task == "detect":
        out["window_auc"] = roc_auc_score(y, p[:, 1]) if len(np.unique(y)) == 2 else np.nan
        out["recording_auc"] = roc_auc_score(rec.y, rec[cols[1]]) if rec.y.nunique() == 2 else np.nan
    return out
