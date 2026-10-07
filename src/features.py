"""Windowing, hand-crafted features and log-spectrograms."""
import numpy as np
from scipy import stats

WINDOW_S = 1.0

# Band edges in Hz. The hydrophone is sampled at 8 kHz and carries almost all
# of its energy below 100 Hz, so it gets finer low-frequency bands.
BANDS = {
    "A": [0, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 12800],
    "P": [0, 20, 50, 100, 200, 500, 1000, 2000, 4000, 8000, 12800],
    "H": [0, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000],
}
N_BANDS = 10
BAND_COLS = [f"band_{i}" for i in range(N_BANDS)]

# Scale-invariant features: unchanged if the whole window is multiplied by a constant.
SHAPE_FEATURES = [
    "kurtosis",
    "skew",
    "crest",
    "zcr",
    "block_cv",
    "f_centroid",
    "f_spread",
    "f_peak",
    "f50",
    "f95",
    "spec_entropy",
    "spec_flatness",
] + BAND_COLS
# The one feature that carries absolute signal level.
AMPLITUDE_FEATURES = ["log_rms"]

SPEC_BANDS = 64
SPEC_FRAMES = 32
_SPEC_NPERSEG = 2048


def split_windows(x, fs, window_s=WINDOW_S):
    """Non-overlapping windows as a 2D array; the trailing partial window is dropped."""
    n = int(round(fs * window_s))
    k = x.size // n
    return x[: k * n].reshape(k, n)


def window_features(w, fs, sensor_type):
    """Features for a 2D array of windows. Returns a dict of 1D arrays."""
    w = w.astype(np.float64)
    w = w - w.mean(axis=1, keepdims=True)
    n = w.shape[1]
    eps = 1e-30
    rms = np.sqrt((w**2).mean(axis=1)) + eps

    out = {
        "log_rms": np.log10(rms),
        "kurtosis": stats.kurtosis(w, axis=1),
        "skew": stats.skew(w, axis=1),
        "crest": np.abs(w).max(axis=1) / rms,
        "zcr": (np.diff(np.signbit(w), axis=1) != 0).mean(axis=1),
    }
    # Stationarity inside the window: spread of RMS over ten sub-blocks.
    blocks = np.sqrt((w[:, : (n // 10) * 10].reshape(w.shape[0], 10, -1) ** 2).mean(axis=2))
    out["block_cv"] = blocks.std(axis=1) / (blocks.mean(axis=1) + eps)

    p = np.abs(np.fft.rfft(w * np.hanning(n), axis=1)) ** 2
    p[:, 0] = 0.0
    f = np.fft.rfftfreq(n, 1.0 / fs)
    total = p.sum(axis=1, keepdims=True) + eps
    q = p / total
    centroid = (q * f).sum(axis=1)
    out["f_centroid"] = centroid
    out["f_spread"] = np.sqrt((q * (f - centroid[:, None]) ** 2).sum(axis=1))
    out["f_peak"] = f[p.argmax(axis=1)]
    c = np.cumsum(q, axis=1)
    out["f50"] = f[(c >= 0.5).argmax(axis=1)]
    out["f95"] = f[(c >= 0.95).argmax(axis=1)]
    out["spec_entropy"] = -(q * np.log(q + eps)).sum(axis=1) / np.log(q.shape[1])
    out["spec_flatness"] = np.exp(np.log(p[:, 1:] + eps).mean(axis=1)) / (p[:, 1:].mean(axis=1) + eps)

    edges = BANDS[sensor_type]
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        mask = (f >= lo) & (f < hi) if i < N_BANDS - 1 else (f >= lo)
        out[f"band_{i}"] = np.log10(q[:, mask].sum(axis=1) + 1e-12)
    return out


def log_spectrogram(w, fs):
    """Log-frequency log-power spectrogram per window, shape (n, SPEC_BANDS, SPEC_FRAMES).

    Each spectrogram is standardised on its own, so it carries spectral shape
    and its change over the window but not absolute level.
    """
    w = w.astype(np.float64)
    w = w - w.mean(axis=1, keepdims=True)
    n = w.shape[1]
    hop = (n - _SPEC_NPERSEG) // (SPEC_FRAMES - 1)
    idx = np.arange(_SPEC_NPERSEG)[None, :] + hop * np.arange(SPEC_FRAMES)[:, None]
    frames = w[:, idx] * np.hanning(_SPEC_NPERSEG)
    p = np.abs(np.fft.rfft(frames, axis=2)) ** 2  # (n, frames, bins)
    f = np.fft.rfftfreq(_SPEC_NPERSEG, 1.0 / fs)
    # Integrate power between log-spaced edges through the cumulative spectrum,
    # so narrow low bands and wide high bands are both handled exactly.
    edges = np.geomspace(f[1], f[-1], SPEC_BANDS + 1)
    cum = np.cumsum(p, axis=2)
    flat = cum.reshape(-1, cum.shape[2])
    at_edges = np.stack([np.interp(edges, f, row) for row in flat]).reshape(p.shape[0], SPEC_FRAMES, -1)
    band = np.diff(at_edges, axis=2) / np.diff(edges)
    s = np.log10(np.maximum(band, 1e-30)).transpose(0, 2, 1)  # (n, bands, frames)
    s = s - s.mean(axis=(1, 2), keepdims=True)
    s = s / (s.std(axis=(1, 2), keepdims=True) + 1e-9)
    return s.astype(np.float16)
