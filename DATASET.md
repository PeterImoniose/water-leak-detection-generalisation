# Water distribution leak detection dataset - data card

Everything in this card under "Measured from the files" was computed from the 282 files in this folder.
Everything under "From the publication" comes from the source paper and Mendeley page and was not independently verified.
Where the two disagree, it is called out in "Discrepancies".

## 1. Source and licence

| Item | Value |
|---|---|
| Title | Dataset of Leak Simulations in Experimental Testbed Water Distribution System |
| Authors | Mohsen Aghashahi, Lina Sela, M. Katherine Banks (Texas A&M University) |
| Paper | "Benchmarking dataset for leak detection and localization in water distribution systems", Data in Brief, 2023, DOI 10.1016/j.dib.2023.109148 |
| Dataset DOI | 10.17632/tbrnp6vrnj.1 (Mendeley Data, published 12 Dec 2022) |
| Licence | CC BY 4.0 - reuse allowed with attribution to the authors |
| Funding | Texas Sea Grant, NSF grant 1943428 |

Links:
- Paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10147960/
- Dataset: https://data.mendeley.com/datasets/tbrnp6vrnj/1

This local copy arrived as a two-part Google Drive download, not directly from Mendeley, so it may differ from the official release (see "Discrepancies").

## 2. From the publication

### Testbed
- Laboratory-scale water distribution network, 7.5 m x 5 m footprint.
- 152.4 mm schedule 80 PVC pipe, 47 m total length, fed by a 25.4 mm supply line.
- Two network layouts: looped and branched.

### Sensors
| Code | Sensor | Model | Sensitivity | Acquisition |
|---|---|---|---|---|
| A1, A2 | Accelerometer | PCB 333B50 | 102 mV/(m/s^2), range +-49 m/s^2 peak | NI-9234, LabVIEW |
| P1, P2 | Dynamic pressure sensor | PCB 102B16 | 7.25 uV/Pa, 1-689 kPa | NI-9234, LabVIEW |
| H1, H2 | Hydrophone | Aquarian H2c | -180 dB re 1 V/uPa | ZOOM UAC-2 audio interface, Audacity |

Sensor placement:
- A1, A2: on the branches of two different tee connections.
- H1, H2: in two hydrants, as far apart as possible and symmetrical about the leak.
- P1: end of the supply line (upstream). P2: farthest corner from the entry point (downstream).
- Exact distances from the leak are not given in the text I could access.

### Leak classes
| Code | Class | Description |
|---|---|---|
| NL | No-leak | Healthy baseline |
| OL | Orifice leak | Hole of about 1.6 mm diameter |
| LC | Longitudinal crack | 2 mm x 1 mm, along the pipe |
| CC | Circumferential crack | 2 mm x 1 mm, around the pipe |
| GL | Gasket leak | Made by loosening a flange's bolts |

### Operating conditions
| Code | Meaning |
|---|---|
| ND | No demand (0 L/s) |
| 0.18 LPS | Steady demand of 0.18 L/s |
| 0.47 LPS | Steady demand of 0.47 L/s |
| Transient | Demand drops from 0.47 L/s to 0 at about 20 s into the recording |
| N / NN | Hydrophone only: with / without added background noise (traffic sound, electric saw) |

## 3. Folder layout and file naming

```
PHME data/
  datasets-20260816T174846Z-1-001/datasets/<Sensor>/<Sensor>/<Topology>/<Leak class>/<file>
  datasets-20260816T174846Z-1-002/datasets/<Sensor>/<Sensor>/<Topology>/<Leak class>/<file>
```

The two parts have the same internal structure and no overlapping files. A loader must read both, or the parts must be merged first.

File name pattern:
- Accelerometer and pressure: `<TOPOLOGY>_<LEAK>_<FLOW>_<SENSOR>.csv`, e.g. `BR_CC_0.18 LPS_A1.csv`
- Hydrophone: `<TOPOLOGY>_<LEAK>_<FLOW>_<NOISE>_<SENSOR>.raw`, e.g. `LO_OL_ND_N_H2.raw`
- Topology is `BR` (branched) or `LO` (looped). Note the space inside `0.18 LPS` and `0.47 LPS`.
- Two extra hydrophone files, `Background Noise_H1.raw` and `Background Noise_H2.raw`, sit outside the grid.

Regex that parses every labelled file:

```
(BR|LO)_(CC|GL|LC|NL|OL)_(0\.18 LPS|0\.47 LPS|ND|Transient)(?:_(NN|N))?_([AHP])([12])
```

## 4. Measured from the files

### Inventory
| Sensor | Files | Size | Format | Sample rate | Duration (min / median / max) |
|---|---|---|---|---|---|
| Accelerometer | 80 | 1.85 GB | CSV | 25.6 kHz | 34.0 / 36.1 / 41.0 s |
| Dynamic pressure | 80 | 1.88 GB | CSV | 25.6 kHz | 35.2 / 36.1 / 41.0 s |
| Hydrophone | 120 + 2 background | 0.15 GB | RAW | 8 kHz | 34.9 / 37.0 / 61.3 s |
| Total | 282 | 3.88 GB | | | |

### Design and class balance
- Accelerometer and pressure: full grid of 2 topologies x 5 leak classes x 4 flow conditions x 2 sensors = 80 files each, 16 files per class.
- Hydrophone: 120 files, 24 per class. Every condition has an `N` recording; only `ND` and `Transient` also have an `NN` recording.
- Each (topology, class, flow) cell is a single recording per sensor. There are no repeats.
- Every file name agrees with the folder it sits in.

### File formats
CSV (accelerometer, pressure):
- Two columns, `Sample,Value`, with a UTF-8 BOM at the start of the file.
- `Sample` is time in seconds, not an index. It is rounded in the text (e.g. `3.91E-05`), so rebuild time as `row / 25600`.
- No missing values and no timing gaps in any of the 160 files.
- About 870,000 to 1,049,000 rows per file.

RAW (hydrophone):
- Headerless signed 32-bit little-endian PCM, mono.
- Only the upper 16 bits carry signal (the lower 16 bits are zero in every sample of every file), so the effective resolution is 16 bits.
- The 8 kHz rate is taken from the publication; the files themselves carry no rate. It is consistent with the data: read at 8 kHz, narrow spectral lines fall at exactly 120 and 240 Hz, the mains harmonics. The 25.6 kHz rate of the CSV files checks out the same way (lines at exactly 60, 180 and 300 Hz).

### Loading

```python
import numpy as np, pandas as pd

def load_csv(path, fs=25600):
    x = pd.read_csv(path, encoding="utf-8-sig")["Value"].to_numpy()
    return np.arange(x.size) / fs, x

def load_hydrophone(path, fs=8000):
    x = (np.fromfile(path, dtype="<i4") >> 16).astype(np.int16)
    return np.arange(x.size) / fs, x
```

### Signal levels
Units are not stated in the files. Values are as stored.

| Sensor | Overall min / max | Median AC RMS | RMS range across files |
|---|---|---|---|
| A1 | -0.226 / 0.236 | 0.0054 | 0.0001 - 0.0142 |
| A2 | -0.216 / 0.238 | 0.0020 | 0.0012 - 0.0036 |
| P1 | -23,010 / 23,743 | 3,042 | 141 - 4,566 |
| P2 | -3,490 / 11,504 | 201 | 20 - 1,278 |
| H1 | -32,768 / 32,767 | 6,008 | 138 - 11,658 |
| H2 | -32,768 / 32,767 | 4,642 | 31 - 10,114 |

### Frequency content
- Accelerometer: most energy is below 2 kHz; median 95% energy frequency is about 2.06 kHz for CC and LC, 1.4 kHz for GL, and about 0.43 kHz for NL and OL. The higher values for CC, LC and GL are driven by the interference described in issue 7 below, not by pipe vibration.
- Pressure P1: 95% of energy below about 400-670 Hz. Pressure P2: 95% below about 22-67 Hz.
- Hydrophone: more than 99.9% of energy is below 100 Hz in every leak class, with the spectral peak at 10-25 Hz. The two background-noise files are 30-140 counts RMS against 4,400-6,100 during tests.

## 5. Data quality issues

1. **Three CSV files are probably truncated.** `BR_LC_0.47 LPS_P1.csv`, `LO_OL_Transient_P2.csv` and `BR_LC_Transient_A1.csv` each have exactly 1,048,572 rows, which is the Excel worksheet row limit less a few rows.
2. **Sensor pairs are not the same length.** None of the 40 A1/A2 pairs or 40 P1/P2 pairs match in length; they differ by up to about 4.9 s. Treat sensors 1 and 2 as not sample-synchronised unless you align them yourself. Hydrophone pairs do match, except `BR_GL_ND_N` (H1 and H2 differ by 3.9 s).
3. **Hydrophone clipping.** 44 of 122 files reach the 16-bit limits; 39 of those are transient recordings. The worst, `LO_LC_Transient_N_H1.raw`, is clipped on 4.0% of samples.
4. **Uneven durations.** Recordings are 34-41 s, and eight hydrophone transient files (branched GL and NL) run 46-61 s. Window or crop to a fixed length before modelling.
5. **Transient recordings are non-stationary.** The P2 transient files contain one large spike at 17.7-21.4 s, which matches the published demand shut-off at about 20 s.
6. **Units are missing** for all three sensors.
7. **Interference in the looped accelerometer recordings.** On A2 in the looped layout, all twelve recordings for LC, CC and GL have the same RMS to within 3% in every flow condition (including the transient), excess kurtosis of about -0.95, and a spectrum that is an evenly spaced comb of narrow lines. They do not respond to flow or to the demand shut-off, so they do not appear to contain pipe vibration. The pattern repeats every 0.22 s (lines about 4.6 Hz apart) and is found in these twelve recordings only: a self-similarity score is 0.83 to 0.86 for them and at most 0.19 for the other 68 accelerometer recordings. They are separate recordings, not copies of one file. On A1 in the looped layout, the LC and CC recordings are roughly 5 to 15 times quieter than NL and OL at every flow condition, and two of the four GL recordings (no demand and 0.47 L/s) sit near the noise floor with 60 Hz mains harmonics. The cause cannot be confirmed from the files. See `notebooks/01_exploratory_analysis.ipynb`, section 1.7.

## 6. Discrepancies with the publication

| Item | Publication | This copy |
|---|---|---|
| Accelerometer / pressure sample rate | 51.2 kHz | Time column steps at 1/25600 s (25.6 kHz) |
| Recording length | 30 s | 34-41 s (up to 61 s for some hydrophone files) |
| File count | 280 | 280 + 2 background-noise files |

The 25.6 kHz time axis is self-consistent: it places the transient spike at about 20 s, as the paper describes. I could not determine whether this copy was downsampled or the paper's figure is nominal.

## 7. Modelling cautions

- **Sensor position and topology dominate amplitude.** P1 is about 15 times louder than P2. A1 on the branched layout sits at 0.0084-0.0090 RMS for every class including no-leak, so raw amplitude carries no leak information there.
- **Class differences are not consistent across layouts.** On looped A2, the 500-2000 Hz band holds about 21% of energy for CC, GL and LC against 0.2% for NL and OL, but that is the interference in issue 7. On branched A2 the separation disappears (0.7-2.3% for all classes). Typical gaps between class-median spectra are 8 to 30 dB in the looped layout on five of six sensors and 2 to 8 dB in the branched layout.
- **Orifice leak resembles no-leak** on the accelerometers in both level and spectrum. Expect it to be the hardest class.
- **Hydrophone noise tag has little visible effect.** `N` and `NN` recordings have similar RMS and spectral centroid.
- **Only one recording per condition.** Cutting a recording into windows and splitting windows randomly between train and test will leak information. Split by whole recording, and ideally hold out a flow condition or a topology.
- **Possible session effects.** Some sensor and layout combinations show one class with a very different spectrum from the rest (e.g. looped P1, where OL has 1% of energy in 500-2000 Hz against about 56% for the other four). With no repeats, a leak effect cannot be separated from a change in setup between recordings.

## 8. Using this with git

- The raw data is 3.88 GB; keep it out of the repository (`.gitignore` the two `datasets-*` folders) and link to the Mendeley DOI instead.
- `file_inventory.csv` in this folder has one row per recording with its labels and summary statistics, and is small enough to commit.
- Cite the paper and dataset DOI in the README to satisfy CC BY 4.0.

## 9. file_inventory.csv columns

| Column | Meaning |
|---|---|
| file, part, sensor_type, bytes | File name, download part (001/002), sensor folder, size |
| topology, leak, flow, noise, sensor | Labels parsed from the file name (`BG` = background noise) |
| n, fs, dur_s | Sample count, sample rate used, duration |
| nan, dt_irregular | Missing values, irregular time steps (CSV only) |
| mean, std, min, max, rms_ac | Basic statistics; `rms_ac` is RMS after removing the mean |
| kurt, crest | Excess kurtosis and crest factor |
| n_unique, frac_at_extreme | Distinct values, share of samples at the min or max value |
| f_peak, f_centroid, f50, f95 | Welch spectrum peak, centroid, and 50% / 95% cumulative energy frequencies (Hz) |
| band_lo_hi | Share of spectral energy in that band (Hz) |
| block_rms_cv, block_rms_maxmin | Variation of 1-second block RMS: coefficient of variation and max/min ratio |
| raw_mod4, low16_nonzero | Hydrophone format checks |
