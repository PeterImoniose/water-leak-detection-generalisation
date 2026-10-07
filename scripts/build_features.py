"""Turn the raw recordings into window features, spectrograms and per-file spectra.

Run from the repository root:  python scripts/build_features.py
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import features as ft  # noqa: E402
import leakdata as ld  # noqa: E402

PSD_NPERSEG = {"A": 8192, "P": 8192, "H": 4096}


def process(row):
    x, fs = ld.load_signal(row["path"])
    w = ft.split_windows(x, fs)
    feats = pd.DataFrame(ft.window_features(w, fs, row["sensor_type"]))
    feats.insert(0, "window", np.arange(len(feats)))
    feats.insert(1, "t_start", feats["window"] * ft.WINDOW_S)
    # Share of samples at the 16-bit limits (hydrophone clipping); zero for CSV sensors.
    clip = ((w >= 32767) | (w <= -32768)).mean(axis=1) if row["sensor_type"] == "H" else np.zeros(len(w))
    feats["clip_frac"] = clip
    for col in ["file", "sensor_type", "sensor", "topology", "leak", "flow", "noise", "condition"]:
        feats.insert(0, col, row[col])
    spec = ft.log_spectrogram(w, fs)
    f, pxx = signal.welch(x - x.mean(), fs=fs, nperseg=PSD_NPERSEG[row["sensor_type"]])
    return feats, spec, pxx.astype(np.float32), f.astype(np.float32)


def main():
    index = ld.build_index()
    print(f"{len(index)} recordings")
    with Pool(8) as pool:
        results = pool.map(process, index.to_dict("records"), chunksize=4)

    ld.PROCESSED.mkdir(parents=True, exist_ok=True)
    feats = pd.concat([r[0] for r in results], ignore_index=True)
    float_cols = feats.select_dtypes("float64").columns
    feats[float_cols] = feats[float_cols].astype(np.float32)
    feats.to_parquet(ld.PROCESSED / "features.parquet", index=False)
    print("features:", feats.shape)

    specs, psd = {}, {}
    for stype in "APH":
        keep = [i for i, t in enumerate(index.sensor_type) if t == stype]
        specs[stype] = np.concatenate([results[i][1] for i in keep])
        psd[f"{stype}_pxx"] = np.stack([results[i][2] for i in keep])
        psd[f"{stype}_f"] = results[keep[0]][3]
        psd[f"{stype}_file"] = index.file.to_numpy()[keep].astype(str)
        # Row order of specs[stype] matches feats[feats.sensor_type == stype].
        assert len(specs[stype]) == (feats.sensor_type == stype).sum()
    np.savez_compressed(ld.PROCESSED / "spectrograms.npz", **specs)
    np.savez_compressed(ld.PROCESSED / "psd.npz", **psd)
    print("done")


if __name__ == "__main__":
    main()
