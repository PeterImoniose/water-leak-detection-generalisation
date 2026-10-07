"""Index and load the water leak detection recordings.

The raw data is not in the repository. Download it from Mendeley
(DOI 10.17632/tbrnp6vrnj.1) and extract it into the repository root, or point
the LEAK_DATA_ROOT environment variable at the folder that contains it.
"""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("LEAK_DATA_ROOT", REPO_ROOT))
PROCESSED = REPO_ROOT / "data" / "processed"

FS = {"A": 25600, "P": 25600, "H": 8000}
SENSOR_NAMES = {"A": "Accelerometer", "P": "Dynamic pressure", "H": "Hydrophone"}
LEAKS = ["NL", "OL", "LC", "CC", "GL"]
LEAK_NAMES = {
    "NL": "No-leak",
    "OL": "Orifice leak",
    "LC": "Longitudinal crack",
    "CC": "Circumferential crack",
    "GL": "Gasket leak",
}
FLOWS = ["ND", "0.18 LPS", "0.47 LPS", "Transient"]
TOPOLOGIES = {"BR": "Branched", "LO": "Looped"}

_NAME = re.compile(
    r"(BR|LO)_(CC|GL|LC|NL|OL)_(0\.18 LPS|0\.47 LPS|ND|Transient)(?:_(NN|N))?_([AHP])([12])$"
)


def build_index(root=DATA_ROOT):
    """One row per recording, with labels parsed from the file name."""
    rows = []
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in (".csv", ".raw") or ".venv" in path.parts:
            continue
        m = _NAME.match(path.stem)
        if m:
            topology, leak, flow, noise, stype, pos = m.groups()
        elif path.stem.startswith("Background Noise"):
            topology, leak, flow, noise, stype, pos = None, "BG", None, None, "H", path.stem[-1]
        else:
            continue
        rows.append(
            dict(
                file=path.name,
                path=str(path),
                sensor_type=stype,
                sensor=f"{stype}{pos}",
                topology=topology,
                leak=leak,
                flow=flow,
                noise=noise or "",
            )
        )
    index = pd.DataFrame(rows)
    if index.empty:
        raise FileNotFoundError(f"No recordings found under {root}. See README.md for the download link.")
    # A condition is one physical test: both sensors of a pair (and the N / NN
    # hydrophone takes) share it, so it is the unit that must not straddle a split.
    index["condition"] = index.topology.fillna("BG") + "|" + index.leak + "|" + index.flow.fillna("BG")
    return index


def load_signal(path):
    """Return (signal as float32, sample rate)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        x = pd.read_csv(path, encoding="utf-8-sig", usecols=["Value"], dtype=np.float32)["Value"].to_numpy()
        return x, FS[path.stem[-2]]
    # Hydrophone: headerless signed 32-bit little-endian PCM, signal in the upper 16 bits.
    x = (np.fromfile(path, dtype="<i4") >> 16).astype(np.float32)
    return x, FS["H"]
