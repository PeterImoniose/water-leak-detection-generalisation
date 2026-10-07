# Water pipe leak detection - sensor data analysis

Analysis of a laboratory water distribution leak dataset: accelerometer, dynamic pressure and hydrophone recordings of four leak types and a no-leak baseline, across two pipe layouts and four flow conditions.

## Contents

| File | What it is |
|---|---|
| [DATASET.md](DATASET.md) | Data card: source, testbed, sensors, file formats, loading code, measured statistics, data quality issues and modelling cautions |
| [file_inventory.csv](file_inventory.csv) | One row per recording (282 rows) with parsed labels and summary statistics |

## Getting the data

The raw data (3.88 GB) is not stored in this repository. Download it from Mendeley Data:

https://data.mendeley.com/datasets/tbrnp6vrnj/1

Place the extracted folders in the repository root. The code expects the layout described in section 3 of [DATASET.md](DATASET.md).

## Dataset credit

Aghashahi, M., Sela, L., Banks, M. K. (2023). Benchmarking dataset for leak detection and localization in water distribution systems. Data in Brief. DOI: 10.1016/j.dib.2023.109148

Dataset DOI: 10.17632/tbrnp6vrnj.1, licensed CC BY 4.0.
