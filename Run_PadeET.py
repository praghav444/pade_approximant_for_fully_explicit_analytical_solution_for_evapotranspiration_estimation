"""
Run_PadeET.py
=============
Runs all four ET models against observed FLUXNET data across all available sites
and writes per-half-hour predictions and per-site error statistics.

Usage
-----
1. Set the data directory paths in the USER CONFIGURATION section below.
2. Run:  python Run_PadeET.py

Output
------
Results/all_sites_predictions.parquet  : all valid half-hourly predictions
Results/per_site_rmse.csv              : per-site RMSE and MBE for all four models

Supported data formats
----------------------
  - FLUXNET Shuttle / AmeriFlux FLUXNET (.zip, FULLSET_HH or FULLSET_HR)
  - AmeriFlux BASE-BADM (.zip, BASE_HH or BASE_HR)
  - FLUXNET2015 (.zip, FLUXNET2015_FULLSET_HH or _HR)
  - ICOS (.parquet)
  - JapanFlux 2024 (.zip, COREVARS_HH)
  - OzFlux TERN (.nc, Level 6)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(__file__))
from MyFuns_ReadFluxData import (read_fluxnet_shuttle, read_fluxnet2015,
                                  read_ICOS, read_JapanFlux2024,
                                  read_AMF_FLUXNET, read_AMF_BASE, read_OzFlux)
from ET_models import (qstar, rho_air, compute_ga, compute_gs,
                        ET_PM, ET_McColl, ET_Pade1, ET_Pade2, CP, LAM, RV)

warnings.filterwarnings('ignore')

# ============================================================
# USER CONFIGURATION
# Set paths to your local data directories for each network.
# Leave the path as an empty string ('') or a non-existent
# directory to skip that network.
# ============================================================
DIR = {
    'Fluxnet_Shuttle'     : '/path/to/FluxNet_Shuttle/data',
    'AMF_FLUXNET_keenan_1': '/path/to/ameriflux_downloads',
    'AMF_FLUXNET_keenan_2': '/path/to/fluxnet_downloads',
    'FLUXNET2015'         : '/path/to/FluxNet2015',
    'ICOS'                : '/path/to/ICOS/data',
    'JapanFlux2024'       : '/path/to/JapanFlux2024/Data',
    'AMF_BASE'            : '/path/to/Raw_Data_AmeriFlux_BASE',
    'AMF_FLUXNET'         : '/path/to/Raw_Data_AmeriFlux_FLUXNET',
    'OzFlux'              : '/path/to/TERN/data',
}

OUT_DIR = os.path.join(os.path.dirname(__file__), 'Results')
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# QC thresholds
# ============================================================
USTAR_MIN    = 0.1     # m/s   u* threshold
LE_QC_MAX    = 1       # keep measured + good gap-filled only
LE_ABS_MAX   = 1200    # W/m2  hard physical cap on observed LE
LE_ABS_MIN   = -150    # W/m2  condensation floor
PRED_CAP_MAX = 2000    # W/m2  cap for model predictions
PRED_CAP_MIN = -500    # W/m2  cap for model predictions
GA_MIN       = 0.001   # m/s   minimum aerodynamic conductance
GA_MAX       = 0.3     # m/s   maximum aerodynamic conductance
GS_MIN       = 1e-4    # m/s   minimum surface conductance
GS_MAX       = 0.15    # m/s   maximum surface conductance
DQ_MIN       = -1      # kg/kg minimum q*(Ts) - qa
EBR_MIN      = 0.8     # energy balance ratio lower bound
EBR_MAX      = 1.2     # energy balance ratio upper bound
EBR_A_MIN    = 50.0    # W/m2  minimum |Rn-G| required before computing EBR
MAX_SITES    = None    # set to an integer for quick test runs

site_info_OzFlux = pd.read_csv(os.path.join(os.path.dirname(__file__), "OzFlux_TERN_site_info.csv"))

# Site catalogue
def _build_site_catalogue():
    priority = {
        'Fluxnet_Shuttle': 1, 'AMF_FLUXNET_keenan_2': 2,
        'AMF_FLUXNET_keenan_1': 3, 'AMF_FLUXNET': 4,
        'ICOS': 5, 'OzFlux': 6, 'JapanFlux2024': 7,
        'FLUXNET2015': 8, 'AMF_BASE': 9,
    }
    ext_map = {
        'Fluxnet_Shuttle': '.zip', 'AMF_FLUXNET_keenan_1': '.zip',
        'AMF_FLUXNET_keenan_2': '.zip', 'FLUXNET2015': '.zip',
        'JapanFlux2024': '.zip', 'AMF_BASE': '.zip',
        'AMF_FLUXNET': '.zip', 'ICOS': '.parquet', 'OzFlux': '_L6.nc',
    }
    records = []
    for src, d in DIR.items():
        if not os.path.isdir(d):
            continue
        ext = ext_map.get(src, '.zip')
        for f in os.listdir(d):
            if not f.endswith(ext):
                continue
            if src == 'OzFlux':
                sid   = f.split('_')[0]
                match = site_info_OzFlux.loc[site_info_OzFlux["Site"] == sid,
                                             "Fluxnet_ID"].values
                sid = match[0] if len(match) > 0 else sid
            else:
                parts = f.split('_')
                sid   = parts[1] if len(parts) > 1 else f
            records.append({'site_id': sid, 'source': src, 'file': f, 'dir': d})

    df = pd.DataFrame(records)
    df['priority'] = df['source'].map(priority)
    df = df.sort_values('priority').drop_duplicates(subset='site_id', keep='first')
    return df.reset_index(drop=True)

# Reader dispatch
def _read_site(source, fname, src_dir):
    if 'BASE-BADM' in fname or source == 'AMF_BASE':
        return read_AMF_BASE(src_dir, fname)
    if source in ('Fluxnet_Shuttle', 'AMF_FLUXNET_keenan_1', 'AMF_FLUXNET_keenan_2'):
        return read_fluxnet_shuttle(src_dir, fname)
    if source == 'FLUXNET2015':
        return read_fluxnet2015(src_dir, fname)
    if source == 'ICOS':
        return read_ICOS(src_dir, fname)
    if source == 'JapanFlux2024':
        return read_JapanFlux2024(src_dir, fname)
    if source == 'AMF_FLUXNET':
        return read_AMF_FLUXNET(src_dir, fname)
    if source == 'OzFlux':
        return read_OzFlux(src_dir, fname)
    return None

# Per-site processing
def process_site(args):
    source, fname, src_dir, site_id = args
    try:
        df = _read_site(source, fname, src_dir)
        if df is None or len(df) == 0:
            return None

        # Ensure all required columns exist
        needed = ['LE_F_MDS', 'LE_F_MDS_QC', 'H_F_MDS', 'NETRAD',
                  'G_F_MDS', 'TA_F', 'PA_F', 'VPD_F', 'WS_F', 'USTAR']
        for c in needed:
            if c not in df.columns:
                df[c] = np.nan
        df.replace(-9999, np.nan, inplace=True)

        # Quality filters
        # 1. LE QC flag
        mask = df['LE_F_MDS_QC'].between(0, LE_QC_MAX)
        # 2. u* threshold
        mask &= df['USTAR'] > USTAR_MIN
        # 3. All required variables present
        for c in ['LE_F_MDS', 'H_F_MDS', 'NETRAD', 'G_F_MDS', 'TA_F', 'PA_F', 'VPD_F', 'WS_F', 'USTAR']:
            mask &= df[c].notna()
        # 4. Physical bounds on observed LE
        mask &= df['LE_F_MDS'].between(LE_ABS_MIN, LE_ABS_MAX)
        # 5. Plausible air temperature and pressure
        mask &= df['TA_F'].between(-40, 55)
        mask &= df['PA_F'] > 50
        # 6. Energy balance ratio - only compute where |Rn-G| is large enough
        #    to avoid numerical instability near zero net radiation
        A_obs = df['NETRAD'] - df['G_F_MDS']
        mask &= A_obs.abs() > EBR_A_MIN
        ebr   = (df['LE_F_MDS'] + df['H_F_MDS']) / A_obs
        mask &= ebr.between(EBR_MIN, EBR_MAX)

        df = df[mask].copy().reset_index(drop=True)
        if len(df) < 50:
            return None

        # Derived variables
        Ta_C   = df['TA_F'].values
        P_kPa  = df['PA_F'].values
        Rn     = df['NETRAD'].values
        G      = df['G_F_MDS'].values
        LE_obs = df['LE_F_MDS'].values
        WS     = df['WS_F'].values
        UST    = df['USTAR'].values

        # Enforce energy balance closure for Ts / gs back-calculation
        H_obs  = Rn - G - LE_obs
        A      = Rn - G

        rho    = rho_air(Ta_C, P_kPa)

        # Specific humidity of air from VPD
        es_a = 0.6108 * np.exp(17.27 * Ta_C / (Ta_C + 237.3))
        ea   = es_a - df['VPD_F'].values
        ea   = np.clip(ea, 1e-6, None)
        qa   = 0.622 * ea / (P_kPa - 0.378 * ea)
        qa   = np.clip(qa, 1e-6, None)

        ga           = compute_ga(WS, UST)
        gs, Ts_C     = compute_gs(LE_obs, H_obs, Ta_C, qa, ga, rho, P_kPa)
        dq           = qstar(Ts_C, P_kPa) - qa
        dT_obs       = H_obs / (rho * CP * ga)   # surface-to-air deltaT [K]

        # Secondary filters: ga, gs, dq must be physically meaningful
        valid = (
            np.isfinite(ga) & np.isfinite(gs) &
            np.isfinite(dq) & np.isfinite(Ts_C) & np.isfinite(Ta_C) &
            (ga > GA_MIN) & (ga < GA_MAX) &
            (gs > GS_MIN) & (gs < GS_MAX) &
            (dq > DQ_MIN)
        )
        if valid.sum() < 50:
            return None

        idx    = np.where(valid)[0]
        Ta_C   = Ta_C[idx];   P_kPa  = P_kPa[idx];  Rn     = Rn[idx]
        G      = G[idx];      LE_obs = LE_obs[idx];  H_obs  = H_obs[idx]
        rho    = rho[idx];    qa     = qa[idx];       ga     = ga[idx]
        gs     = gs[idx];     Ts_C   = Ts_C[idx];    dT_obs = dT_obs[idx]
        A      = A[idx]
        dt_sub = df['DateTime'].iloc[idx].values

        # Run all four models
        kw = dict(P_kPa=P_kPa, rho=rho)
        lE_PM                   = ET_PM    (Rn, G, Ta_C, qa, ga, gs, **kw)
        lE_McColl               = ET_McColl(Rn, G, Ta_C, qa, ga, gs, **kw)
        lE_Pade1, x_pade1       = ET_Pade1 (Rn, G, Ta_C, qa, ga, gs, **kw)
        lE_Pade2, x_pade2       = ET_Pade2 (Rn, G, Ta_C, qa, ga, gs, **kw)

        # NaN-ify physically implausible predictions
        n_before = len(lE_PM)
        for arr in (lE_PM, lE_McColl, lE_Pade1, lE_Pade2):
            arr[(arr < PRED_CAP_MIN) | (arr > PRED_CAP_MAX)] = np.nan

        # Joint valid mask: require ALL models finite so RMSE comparisons are
        # always on identical observation subsets (like-for-like)
        ok        = (np.isfinite(lE_PM) & np.isfinite(lE_McColl) &
                     np.isfinite(lE_Pade1) & np.isfinite(lE_Pade2))
        n_dropped = int(n_before - ok.sum())
        if ok.sum() < 50:
            return None

        # Build output dataframe
        out = pd.DataFrame({
            'DateTime'  : dt_sub[ok],
            'site_id'   : site_id,
            'LE_obs'    : LE_obs[ok],
            'Rn_G'      : A[ok],
            'Rn'        : Rn[ok],
            'G'         : G[ok],
            'Ta_C'      : Ta_C[ok],
            'P_kPa'     : P_kPa[ok],
            'qa'        : qa[ok],
            'ga'        : ga[ok],
            'gs'        : gs[ok],
            'rho_a'     : rho[ok],
            'dT_obs'    : dT_obs[ok],
            'lE_PM'     : lE_PM[ok],
            'lE_McColl' : lE_McColl[ok],
            'lE_Pade1'  : lE_Pade1[ok],
            'lE_Pade2'  : lE_Pade2[ok],
            'x_pade1'   : x_pade1[ok],
            'x_pade2'   : x_pade2[ok],
        })

        # Per-site statistics
        def _rmse(pred, obs): return np.sqrt(np.mean((pred - obs)**2))
        def _mbe (pred, obs): return np.mean(pred - obs)

        obs_v = out['LE_obs'].values
        x1_ok = out['x_pade1'].values
        x2_ok = out['x_pade2'].values

        # Mask for periods where both Pade approximations are in their accurate regime
        pade_mask = (np.abs(x1_ok) < 1.0) & (np.abs(x2_ok) < 1.0)
        out_pade  = out[pade_mask]
        obs_pade  = out_pade['LE_obs'].values

        rmse_row = {
            'site_id'       : site_id,
            'source'        : source,
            'n_obs'         : len(out),
            'n_dropped'     : n_dropped,
            'n_pade_valid'  : int(pade_mask.sum()),
            'RMSE_PM'       : _rmse(out_pade['lE_PM'].values,     obs_pade),
            'RMSE_McColl'   : _rmse(out_pade['lE_McColl'].values, obs_pade),
            'RMSE_Pade1'    : _rmse(out_pade['lE_Pade1'].values,  obs_pade),
            'RMSE_Pade2'    : _rmse(out_pade['lE_Pade2'].values,  obs_pade),
            'MBE_PM'        : _mbe (out_pade['lE_PM'].values,     obs_pade),
            'MBE_McColl'    : _mbe (out_pade['lE_McColl'].values, obs_pade),
            'MBE_Pade1'     : _mbe (out_pade['lE_Pade1'].values,  obs_pade),
            'MBE_Pade2'     : _mbe (out_pade['lE_Pade2'].values,  obs_pade),
            'dT_median'     : float(np.median(np.abs(dT_obs[ok]))),
            'dT_p95'        : float(np.percentile(np.abs(dT_obs[ok]), 95)),
            'frac_xp1_gt1'  : float(np.nanmean(x1_ok > 1.0)),
            'frac_xp1_gt2'  : float(np.nanmean(x1_ok > 2.0)),
            'frac_xp2_gt1'  : float(np.nanmean(x2_ok > 1.0)),
            'frac_xp2_gt2'  : float(np.nanmean(x2_ok > 2.0)),
        }
        drop_str = f"  drop={n_dropped}" if n_dropped > 0 else ""
        print(f"  {site_id:12s}  n={len(out):6d}  n_valid={pade_mask.sum():6d}{drop_str}  "
              f"RMSE PM={rmse_row['RMSE_PM']:.2f}  "
              f"McColl={rmse_row['RMSE_McColl']:.2f}  "
              f"Pade1={rmse_row['RMSE_Pade1']:.2f}  "
              f"Pade2={rmse_row['RMSE_Pade2']:.2f}", flush=True)
        return out, rmse_row
    except Exception as e:
        print(f"  ERROR {site_id}: {e}", flush=True)
        return None

# Main
if __name__ == '__main__':
    t0 = time.time()

    catalogue = _build_site_catalogue()
    print(f"Total unique sites: {len(catalogue)}")
    catalogue.iloc[:, :2].to_csv(os.path.join(os.path.dirname(__file__), "site_catalogue.csv"), index=False)
    if MAX_SITES is not None:
        catalogue = catalogue.head(MAX_SITES)
        print(f"[TEST MODE] Limited to first {MAX_SITES} sites")

    tasks = [(row.source, row.file, row.dir, row.site_id)
             for row in catalogue.itertuples()]

    n_workers = min(128, cpu_count())
    print(f"Running with {n_workers} workers\n")

    with Pool(n_workers) as pool:
        results = pool.map(process_site, tasks)

    all_preds = [r[0] for r in results if r is not None]
    all_rmse  = [r[1] for r in results if r is not None]

    if not all_preds:
        print("No valid sites processed.")
        sys.exit(1)

    df_all  = pd.concat(all_preds,  ignore_index=True)
    df_rmse = pd.DataFrame(all_rmse)

    df_all.to_parquet(os.path.join(OUT_DIR, 'all_sites_predictions.parquet'), index=False)
    df_rmse.to_csv(os.path.join(OUT_DIR, 'per_site_rmse.csv'), index=False)

    obs = df_all['LE_obs'].values
    def _rmse(col): return np.sqrt(np.mean((df_all[col].values - obs)**2))
    def _mbe (col): return np.mean(df_all[col].values - obs)

    print(f"\n{'='*60}")
    print(f"Sites with valid data : {len(df_rmse)}")
    print(f"Total observations    : {len(df_all):,}")
    print(f"\n{'Model':<14} {'RMSE':>8} {'MBE':>9}  [W m-2]")
    print(f"{'-'*35}")
    for col, label in [('lE_PM','PM'), ('lE_McColl','McColl'),
                       ('lE_Pade1','Pade-1'), ('lE_Pade2','Pade-2')]:
        print(f"  {label:<12} {_rmse(col):>8.3f} {_mbe(col):>+9.3f}")

    day = df_all['Rn_G'] > 0
    print(f"\n{'Model':<14} {'Day RMSE':>10} {'Night RMSE':>12}  [W m-2]")
    print(f"{'-'*40}")
    def _rmse_sub(col, mask):
        d = df_all[mask]
        return np.sqrt(np.mean((d[col].values - d['LE_obs'].values)**2))
    for col, label in [('lE_PM','PM'), ('lE_McColl','McColl'),
                       ('lE_Pade1','Pade-1'), ('lE_Pade2','Pade-2')]:
        print(f"  {label:<12} {_rmse_sub(col, day):>10.3f} "
              f"{_rmse_sub(col, ~day):>12.3f}")

    print(f"\nTotal runtime: {(time.time()-t0)/60:.1f} min")
    print(f"Results saved to {OUT_DIR}")
