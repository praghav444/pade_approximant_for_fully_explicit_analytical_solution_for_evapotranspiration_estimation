
# Source Code: Pade Approximation of Clausius-Clapeyron for Evapotranspiration Estimation
This repository contains the Python source code and example data accompanying the manuscript:

> Raghav, P. & Kumar, M. (2026). Beyond Penman-Monteith: A More Accurate Fully Explicit Analytical Solution for Evapotranspiration Estimation.

> Contact: Pushpendra Raghav; https://praghav444.github.io/ | ppushpendra@ua.edu | praghav444@gmail.com | University of Alabama 
#-----------------------------------------------------------------------------------------------------------

## Overview

This code implements and evaluates four analytical solutions for the evapotranspiration (ET) from the Penman-Monteith framework, each using a different approximation of the saturation specific humidity function (Clausius-Clapeyron relation):

| Model | Approximation | Reference |
|-------|--------------|-----------|
| Penman-Monteith (PM) | Linearised (1st-order Taylor) | Penman (1948), Monteith (1965); https://doi.org/10.1098/rspa.1948.0037|
| McColl | Vallis exponential + Lambert-W | McColl (2020); https://doi.org/10.1029/2020WR027106|
| Solution-1 (Pade-1) | Pade [1,1] of Vallis exponential | This study |
| Solution-2 (Pade-2) | Pade [1,1] of exact Clausius-Clapeyron | This study |

The models were evaluated against native (half-hourly or hourly) eddy-covariance observations from over 600 FLUXNET sites worldwide.

---

## Files

| File | Description |
|------|-------------|
| `ET_models.py` | Core implementation of all four ET models and helper functions |
| `Run_PadeET.py` | Full multi-site pipeline: reads FLUXNET data, applies QC, runs models, saves results |
| `MyFuns_ReadFluxData.py` | Data readers for multiple FLUXNET data formats (Shuttle, FLUXNET2015, ICOS, AmeriFlux, OzFlux, JapanFlux) |
| `meteo_utils.py` | Meteorological utility functions (vapour pressure, psychrometric constant, etc.) |
| `run_example.py` | Standalone example script: runs all four models on the included US-MMS data |
| `OzFlux_TERN_site_info.csv` | Site ID lookup table for OzFlux/TERN sites |
| `example_data/US-MMS_example_input.csv` | Example input data: Morgan Monroe State Forest, Indiana, USA (9,276 half-hourly time steps) |

### Output data (uploaded separately)

| File | Description |
|------|-------------|
| `Results/all_sites_predictions.parquet` | Half-hourly predictions from all four models across 600 FLUXNET sites (5 million+ time steps) |

---

## Quick Start: Run the Example

The fastest way to verify that the code works is to run the self-contained example script. It uses the included US-MMS data file and only requires `ET_models.py`.

**Step 1:** Install dependencies (see Requirements below).

**Step 2:** From the directory containing the code files, run:

```
python run_example.py
```
---

## Run the Full Multi-Site Pipeline

To reproduce the full analysis across all FLUXNET sites:

**Step 1:** Edit `Run_PadeET.py` and set the directory paths in the `USER CONFIGURATION` section at the top of the file to point to your local FLUXNET data directories.

**Step 2:** Run:

```
python Run_PadeET.py
```

Output files are written to a `Results/` folder.

---

## Input Data (US-MMS Example)

The file `example_data/US-MMS_example_input.csv` contains quality-controlled,
half-hourly meteorological and flux inputs for the Morgan Monroe State Forest
site (AmeriFlux site ID: US-MMS), a temperate deciduous forest in Indiana, USA.

These inputs were derived from the FLUXNET data product following the
preprocessing steps described in the Methods section of the paper. The file
includes the following columns:

| Column | Units | Description |
|--------|-------|-------------|
| DateTime | - | Timestamp (YYYY-MM-DD HH:MM:SS) |
| site_id | - | FLUXNET site identifier |
| LE_obs | W m-2 | Observed latent heat flux |
| Rn | W m-2 | Net radiation |
| G | W m-2 | Ground heat flux |
| Ta_C | deg C | Air temperature |
| P_kPa | kPa | Atmospheric pressure |
| qa | kg kg-1 | Specific humidity of air |
| ga | m s-1 | Aerodynamic conductance |
| gs | m s-1 | Surface conductance |
| rho_a | kg m-3 | Air density |
| dT_obs | K | Observed surface-to-air temperature difference |

---

## Requirements

- Python 3.8 or later
- numpy
- scipy
- pandas
- pyarrow (for reading/writing .parquet files)
- matplotlib (for figure generation only)
- xarray (for OzFlux data reading only)

Install all dependencies with:

```
pip install numpy scipy pandas pyarrow matplotlib xarray
```

---

## Notes on Data Availability

The raw FLUXNET eddy-covariance data used in this study are available from the following sources:

- FLUXNET2015: https://fluxnet.org/data/fluxnet2015-dataset/
- AmeriFlux: https://ameriflux.lbl.gov/
- ICOS: https://www.icos-cp.eu/
- OzFlux/TERN: https://www.tern.org.au/
- JapanFlux: https://ads.nipr.ac.jp/japan-flux2024/

FLUXNET Shuttle Library (https://github.com/fluxnet/shuttle) from the FLUXNET Shuttle (one stop for accessing eddy covariance data; https://data.fluxnet.org/) can also be used to directly access data from different sources.

The FLUXNET data are shared under a CC-BY-4.0 data use license which requires attribution for each data use.
---

## License

The source code is released under the MIT License. See LICENSE for details.

---

## Contact

For questions about the code, contact me at ppushpendra@ua.edu (or praghav444@gmail.com) or the corresponding author at the email address provided in the manuscript.
