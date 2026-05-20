# -*- coding: utf-8 -*-
import os; import re
import zipfile
import pandas as pd; import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from meteo_utils import calculate_VPD_from_RH, calculate_RH_from_VPD
#--------------------------------------------------------------------------------------------
def flux_c(LE, H, Rn, G):
    LE = np.asarray(LE, dtype=float)
    H  = np.asarray(H, dtype=float)
    Rn = np.asarray(Rn, dtype=float)
    G  = np.asarray(G, dtype=float)
    res = Rn - G - LE - H
    denom = LE + H
    denom[np.abs(denom) < 1e-6] = np.nan
    Br = LE / denom
    Br = np.clip(Br, -5, 5)
    LE_f = LE + res * Br
    H_f  = H + res * (1 - Br)
    return LE_f, H_f
def select_columns(df, prefix, preferred_contains='PI', exclude_suffix='QC'):
    exact_cols = [c for c in df.columns if c == prefix and not c.endswith(exclude_suffix)]
    exact_preferred = [c for c in exact_cols if preferred_contains in c]
    if exact_preferred:
        return exact_preferred
    if exact_cols:
        return exact_cols
    prefixed_cols = [c for c in df.columns
                     if c.startswith(prefix + '_') and not c.endswith(exclude_suffix)]
    if not prefixed_cols:
        return []
    preferred_cols = [c for c in prefixed_cols if preferred_contains in c]
    if preferred_cols:
        return preferred_cols
    # detect AmeriFlux tower variables VAR_H_V_R
    tower_cols = []
    for c in prefixed_cols:
        m = re.match(rf"{prefix}_(\d+)_(\d+)_(\d+)", c)
        if m:
            H, V, R = map(int, m.groups())
            if V == 1:   # top-of-canopy
                tower_cols.append(c)
    if tower_cols:
        return tower_cols
    return prefixed_cols
#--------------------------------------------------------------------------------------------
# Read Fluxnet Shuttle Data
def read_fluxnet_shuttle(dir_1, zip_file):
    source = zip_file.split("_")[0]
    site_name = zip_file.split("_")[1]
    zip_path = os.path.join(dir_1, zip_file)
    def extract_from_zip(zip_ref):
        target_files = [
            info for info in zip_ref.infolist()
            if ("FLUXNET_FULLSET_HH" in info.filename or "FLUXNET_FULLSET_HR" in info.filename 
                or "FLUXNET2015_FULLSET_HH" in info.filename or "FLUXNET2015_FULLSET_HR" in info.filename 
                or "FLUXNET_FLUXMET_HH" in info.filename or "FLUXNET_FLUXMET_HR" in info.filename)]
        if not target_files:
            return None
        target_file = target_files[0].filename
        with zip_ref.open(target_file) as f:
            df = pd.read_csv(f)
        return df
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            df = extract_from_zip(zip_ref)
            if df is None:
                for file in zip_ref.infolist():
                    if file.filename.endswith(".zip"):
                        with zip_ref.open(file.filename) as subzip_file:
                            with zipfile.ZipFile(subzip_file) as subzip_ref:
                                df = extract_from_zip(subzip_ref)
                                if df is not None:
                                    print(f"Found data in nested zip file: {file.filename}")
                                    break
    except zipfile.BadZipFile:
        print(f"Error: {zip_file} is not a valid zip file.")
        return None
    except FileNotFoundError:
        print(f"Error: {zip_file} not found in the directory.")
        return None
    except Exception as e:
        print(f"Error while processing {zip_file}: {e}")
        return None
    if df is None:
        print(f"No FLUXNET file found for {site_name} | Source: FLUXNET_SHUTTLE")
        return None

    df.replace(-9999, np.nan, inplace=True)
    df['DateTime'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")

    swc_cols = [c for c in df.columns if c.startswith("SWC") and not c.endswith("_QC")]
    if len(swc_cols) == 0:
        theta_1 = np.full(len(df), np.nan)
        theta_2 = np.full(len(df), np.nan)
    else:
        theta_1 = df[swc_cols[0]] / 100            # Surface Soil Moisture (m3 m-3)
        theta_2 = df[swc_cols].mean(axis=1) / 100  # Effective Soil Moisture (m3 m-3)

    NETRAD_cols = select_columns(df, 'NETRAD')
    if NETRAD_cols:
        df['NETRAD'] = df[NETRAD_cols].mean(axis=1)
    else:
        # Compute NETRAD from incoming and outgoing radiation if not already found
        SW_IN_cols = select_columns(df, 'SW_IN')
        SW_OUT_cols = select_columns(df, 'SW_OUT')
        LW_IN_cols = select_columns(df, 'LW_IN')
        LW_OUT_cols = select_columns(df, 'LW_OUT')
        if all(len(cols) > 0 for cols in [SW_IN_cols, SW_OUT_cols, LW_IN_cols, LW_OUT_cols]):
            SW_IN = df[SW_IN_cols].mean(axis=1)
            SW_OUT = df[SW_OUT_cols].mean(axis=1)
            LW_IN = df[LW_IN_cols].mean(axis=1)
            LW_OUT = df[LW_OUT_cols].mean(axis=1)
            df['NETRAD'] = SW_IN - SW_OUT + LW_IN - LW_OUT
    G_cols = select_columns(df, 'G')
    df['G_F_MDS'] = df[G_cols].mean(axis=1) if G_cols else np.nan

    required_columns = ['LE_F_MDS', 'TA_F', 'P_F', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS',
                        'H_F_MDS_QC', 'H_CORR', 'NETRAD', 'G_F_MDS', 'PA_F', 'VPD_F',
                        'GPP_NT_VUT_REF', 'WS_F', 'USTAR']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Compute OFC fallback if LE_CORR is not provided by the file
    if df['LE_CORR'].isna().all():
        _G  = df['G_F_MDS'].fillna(0).values
        _Rn = df['NETRAD'].values.copy()
        _LE = df['LE_F_MDS'].values
        _H  = df['H_F_MDS'].values
        _mask = np.isnan(_Rn)
        _Rn[_mask] = _LE[_mask] + _H[_mask]
        _lec, _hec = flux_c(_LE, _H, _Rn, _G)
        df['LE_CORR'] = df['LE_CORR'].fillna(pd.Series(_lec, index=df.index))
        df['H_CORR']  = df['H_CORR'].fillna(pd.Series(_hec,  index=df.index))

    df = pd.concat([df, pd.DataFrame({'theta_1': theta_1, 'theta_2': theta_2})], axis=1)
    df = df[['DateTime', 'LE_F_MDS', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS', 'H_F_MDS_QC', 'H_CORR', 'NETRAD',
             'G_F_MDS', 'TA_F', 'P_F', 'PA_F', 'VPD_F', 'theta_1', 'theta_2', 'GPP_NT_VUT_REF',
             'WS_F', 'USTAR']]

    df.loc[df['PA_F'] > 400, 'PA_F'] /= 1000  # kPa
    df['VPD_F'] = df['VPD_F'] / 10            # from hPa to kPa
    return df
#--------------------------------------------------------------------------------------------
# Read JapanFlux2024 Data
def read_JapanFlux2024(dir_1,zip_name):
    try:
        zip_path = os.path.join(dir_1, zip_name)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            df = None
            for file in zf.namelist():
                if "COREVARS/" in file.replace("\\", "/") and "COREVARS_HH" in os.path.basename(file):
                    with zf.open(file) as csv_file:
                        site_name = file.split('/')[-1].split('_')[1]
                        match = re.search(r"_(\d{4})-(\d{4})_", file)
                        if match:
                            start_year = int(match.group(1))
                            end_year = int(match.group(2))
                        df = pd.read_csv(csv_file)
            if df is None:
                print("No COREVARS_HH file found in zip")
                return None
        df.replace(-9999, np.nan, inplace=True)
        df['DateTime'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M", errors="coerce")
        df = df.set_index('DateTime')
        time_diffs = df.index.to_series().diff().dropna()
        mode_freq = time_diffs.mode()[0]
        print(mode_freq)
        if mode_freq == pd.Timedelta(minutes=30):
            time_stamp = "HH"
        elif mode_freq == pd.Timedelta(hours=1):
            time_stamp = "HR"
        else:
            time_stamp = "UNKNOWN"
        swc_cols = [col for col in df.columns if col.startswith("SWC") and not col.endswith("_QC")]     
        if len(swc_cols) == 0:
            theta_1 = np.full(len(df), np.nan)
            theta_2 = np.full(len(df), np.nan)
        else:
            df['theta_1'] = df[swc_cols[0]].copy()                # Surface Soil Moisture (m3 m-3)
            df.loc[df['theta_1'] > 1, 'theta_1'] /= 100
            df['theta_2'] = df[swc_cols].mean(axis=1)             # Effective Soil Moisture across all available sensors (m3 m-3)
            df.loc[df['theta_2'] > 1, 'theta_2'] /= 100
        qc_vars = ['LE_F_MDS_QC', 'H_F_MDS_QC', 'NETRAD_F_MDS_QC', 'G_F_MDS_QC']
        for var in qc_vars:
            if var not in df.columns:
                df[var] = 1
        if 'G_F_MDS' not in df.columns:
            df['G_F_MDS'] = np.nan
        df['G_F_MDS'] = df['G_F_MDS'].fillna(0)
        df[qc_vars] = df[qc_vars].fillna(1)
        # Energy Balance Corrections (Optional)
        LE = df['LE_F_MDS'].values; H = df['H_F_MDS'].values   
        df['NETRAD_F_MDS'] = df['NETRAD_F_MDS'].fillna(df['LE_F_MDS'] + df['H_F_MDS'])
        Rn = df['NETRAD_F_MDS'].values; G = df['G_F_MDS'].values
        LE_CORR, H_CORR = flux_c(LE, H, Rn, G)  # Corrected LE and H       <-----
        df['LE_CORR'], df['H_CORR'] = LE_CORR, H_CORR
        df = df.rename(columns={'NETRAD_F_MDS': 'NETRAD', 'GPP_NT_vUT_USTAR50':'GPP_NT_VUT_REF'})
        df = df.reset_index()
        required_columns = ['LE_F_MDS', 'TA_F', 'P_F', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS',
                            'H_F_MDS_QC', 'H_CORR', 'NETRAD', 'G_F_MDS', 'PA_F', 'VPD_F',
                            'GPP_NT_VUT_REF', 'WS_F', 'USTAR']
        for col in required_columns:
            if col not in df.columns:
                df[col] = np.nan
        df = df[['DateTime', 'LE_F_MDS', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS', 'H_F_MDS_QC', 'H_CORR', 'NETRAD', 'G_F_MDS',
                 'TA_F', 'P_F', 'PA_F', 'VPD_F', 'theta_1', 'theta_2', 'GPP_NT_VUT_REF',
                 'WS_F', 'USTAR']]
        df.loc[df['PA_F'] > 400, 'PA_F'] /= 1000  # kPa
        df['VPD_F'] = df['VPD_F']/10              # hPa to kPa
        return df
    except Exception as e:
        print(e)
        return None
#--------------------------------------------------------------------------------------------
# Read OzFlux Data
def read_OzFlux(nc_dir,nc_file):
    try:
        df = xr.open_dataset(os.path.join(nc_dir, nc_file))
        df = df.where(df != -9999, np.nan)
        
        DateTime = pd.to_datetime(df.time.values, format="mixed")
        
        qc_vars = ['Fe_QCFlag', 'Fh_QCFlag', 'Fsd_QCFlag', 'Fg_QCFlag']
        df[qc_vars] = df[qc_vars].fillna(1)
        
        EVI = df.EVI.values.squeeze()                 # Enhanced Vegetation Index (-)
        SRad = df.Fsd.values.squeeze()                # Incoming Solar Radiation (W m-2)
        SRad_QC = df.Fsd_QCFlag.values.squeeze()      # Incoming Solar Radiation (W m-2)
        PA = df.ps.values.squeeze()                   # Surface Air Pressure (kPa)  
        Ta = df.Ta.values.squeeze()                   # Air Temperature (degC)  
    
        LE = df.Fe.values.squeeze()                   # Latent Heat Flux (W m-2)
        LE_QC = df.Fe_QCFlag.values.squeeze()         # Quality Flag for LE (0 and 1 are good quality data)
        H = df.Fh.values.squeeze()                    # Sensible Heat Flux (W m-2)
        H_QC = df.Fh_QCFlag.values.squeeze()          # Quality Flag for H (0 and 1 are good quality data)
        Rn = df.Fn.values.squeeze()                   # Net Radiation (W m-2)
        G = df.Fg.values.squeeze()                    # Ground Heat Flux (W m-2)      
        G_QC = df.Fg_QCFlag.values.squeeze()          

        G[np.isnan(G)] = 0 
        mask = np.isnan(Rn)
        Rn[mask] = LE[mask] + H[mask] 
        # Energy Balance Corrections (Optional)
        LE_CORR, H_CORR = flux_c(LE, H, Rn, G)
        ws  = df['Ws'].values.squeeze()  if 'Ws'    in df else np.full(len(DateTime), np.nan)
        ust = df['ustar'].values.squeeze() if 'ustar' in df else np.full(len(DateTime), np.nan)
        df = pd.DataFrame({'DateTime':DateTime, 'LE_F_MDS':LE, 'LE_F_MDS_QC':LE_QC, 'LE_CORR':LE_CORR, 'H_F_MDS':H, 'H_F_MDS_QC':H_QC, 'H_CORR':H_CORR,
                           'NETRAD': Rn, 'G_F_MDS': G, 'SW_IN_F': SRad,
                           'TA_F': Ta, 'P_F':df.Precip.values.squeeze(), 'PA_F':PA, 'VPD_F':df.VPD.values.squeeze(),
                           'theta_1':df.Sws.values.squeeze(), 'theta_2':df.Sws.values.squeeze(),
                           'GPP_NT_VUT_REF': np.full(len(DateTime), np.nan),
                           'WS_F': ws, 'USTAR': ust})
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        return df
    except Exception as e:
        print(e)
        return None
#--------------------------------------------------------------------------------------------
# Read AmeriFlux-Fluxnet Data
def read_AMF_FLUXNET(dir_1, zip_file):
    source = zip_file.split("_")[0]
    site_name = zip_file.split("_")[1]
    zip_path = os.path.join(dir_1, zip_file)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            target_files = [
                info for info in zip_ref.infolist()
                if ("FLUXNET_FULLSET_HH" in info.filename or "FLUXNET_FULLSET_HR" in info.filename 
                    or "FLUXNET_FLUXMET_HH" in info.filename or "FLUXNET_FLUXMET_HR" in info.filename)]         
            if not target_files:
                print(f"No FLUXNET file found for {site_name}  | Source: AMF_FLUXNET")
                return None
            target_file = target_files[0].filename
            with zip_ref.open(target_file) as f:
                df = pd.read_csv(f)     
    except zipfile.BadZipFile:
        print(f"Error: {zip_file} is not a valid zip file.")
        return None
    except FileNotFoundError:
        print(f"Error: {zip_file} not found in the directory.")
        return None
    except Exception as e:
        print(f"Error while processing {zip_file}: {e}")
        return None
    df.replace(-9999, np.nan, inplace=True)
    df['DateTime'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")

    swc_cols = [c for c in df.columns if c.startswith("SWC") and not c.endswith("_QC")]
    if len(swc_cols) == 0:
        theta_1 = np.full(len(df), np.nan)
        theta_2 = np.full(len(df), np.nan)
    else:
        theta_1 = df[swc_cols[0]] / 100            # Surface Soil Moisture (m3 m-3)
        theta_2 = df[swc_cols].mean(axis=1) / 100  # Effective Soil Moisture (m3 m-3)

    NETRAD_cols = select_columns(df, 'NETRAD')
    if NETRAD_cols:
        df['NETRAD'] = df[NETRAD_cols].mean(axis=1)
    else:
        # Compute NETRAD from incoming and outgoing radiation if not already found
        SW_IN_cols = select_columns(df, 'SW_IN')
        SW_OUT_cols = select_columns(df, 'SW_OUT')
        LW_IN_cols = select_columns(df, 'LW_IN')
        LW_OUT_cols = select_columns(df, 'LW_OUT')
        if all(len(cols) > 0 for cols in [SW_IN_cols, SW_OUT_cols, LW_IN_cols, LW_OUT_cols]):
            SW_IN = df[SW_IN_cols].mean(axis=1)
            SW_OUT = df[SW_OUT_cols].mean(axis=1)
            LW_IN = df[LW_IN_cols].mean(axis=1)
            LW_OUT = df[LW_OUT_cols].mean(axis=1)
            df['NETRAD'] = SW_IN - SW_OUT + LW_IN - LW_OUT
    G = df['G_F_MDS'].values
    Rn = df['NETRAD'].values
    LE = df['LE_F_MDS'].values
    H = df['H_F_MDS'].values
    G[np.isnan(G)] = 0 
    mask = np.isnan(Rn)
    Rn[mask] = LE[mask] + H[mask]
    LE_CORR_back, H_CORR_back = flux_c(LE, H, Rn, G)
    df['LE_CORR'] = df.get('LE_CORR', pd.Series(LE_CORR_back, index=df.index)).fillna(pd.Series(LE_CORR_back, index=df.index))
    df['H_CORR'] = df.get('H_CORR', pd.Series(H_CORR_back, index=df.index)).fillna(pd.Series(H_CORR_back, index=df.index))
    required_columns = ['LE_F_MDS', 'TA_F', 'P_F', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS',
                        'H_F_MDS_QC', 'H_CORR', 'NETRAD', 'G_F_MDS', 'PA_F', 'VPD_F',
                        'GPP_NT_VUT_REF', 'WS_F', 'USTAR']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = pd.concat([df, pd.DataFrame({'theta_1': theta_1, 'theta_2': theta_2})], axis=1)
    df = df[['DateTime', 'LE_F_MDS', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS', 'H_F_MDS_QC', 'H_CORR', 'NETRAD',
             'G_F_MDS', 'TA_F', 'P_F', 'PA_F', 'VPD_F', 'theta_1', 'theta_2', 'GPP_NT_VUT_REF',
             'WS_F', 'USTAR']]
    df.loc[df['PA_F'] > 400, 'PA_F'] /= 1000  # from Pa to kPa
    df['VPD_F'] = df['VPD_F'] / 10            # from hPa to kPa
    return df
#--------------------------------------------------------------------------------------------
# Read AmeriFlux-Base Data
def read_AMF_BASE(dir_1, zip_file):
    source = zip_file.split("_")[0]
    site_name = zip_file.split("_")[1]
    zip_path = os.path.join(dir_1, zip_file)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            target_files = [
                info for info in zip_ref.infolist()
                if ("BASE_HH" in info.filename or "BASE_HR" in info.filename)]
            if not target_files:
                print(f"No FLUXNET file found for {site_name} | Source: AMF_BASE")
                return None
            target_file = target_files[0].filename
            with zip_ref.open(target_file) as f:
                df = pd.read_csv(f, skiprows=2)
    except zipfile.BadZipFile:
        print(f"Error: {zip_file} is not a valid zip file.")
        return None
    except FileNotFoundError:
        print(f"Error: {zip_file} not found in the directory.")
        return None
    except Exception as e:
        print(f"Error while processing {zip_file}: {e}")
        return None
    df.replace(-9999, np.nan, inplace=True)
    df['DateTime'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")

    swc_cols = select_columns(df, 'SWC')
    if len(swc_cols) == 0:
        theta_1 = np.full(len(df), np.nan)
        theta_2 = np.full(len(df), np.nan)
    else:
        theta_1 = df[swc_cols[0]] / 100            # Surface Soil Moisture (m3 m-3)
        theta_2 = df[swc_cols].mean(axis=1) / 100  # Effective Soil Moisture (m3 m-3)

    NETRAD_cols = select_columns(df, 'NETRAD')
    if NETRAD_cols:
        df['NETRAD'] = df[NETRAD_cols].mean(axis=1)
    else:
        # Compute NETRAD from incoming and outgoing radiation if not already found
        SW_IN_cols = select_columns(df, 'SW_IN')
        SW_OUT_cols = select_columns(df, 'SW_OUT')
        LW_IN_cols = select_columns(df, 'LW_IN')
        LW_OUT_cols = select_columns(df, 'LW_OUT')
        if all(len(cols) > 0 for cols in [SW_IN_cols, SW_OUT_cols, LW_IN_cols, LW_OUT_cols]):
            SW_IN = df[SW_IN_cols].mean(axis=1)
            SW_OUT = df[SW_OUT_cols].mean(axis=1)
            LW_IN = df[LW_IN_cols].mean(axis=1)
            LW_OUT = df[LW_OUT_cols].mean(axis=1)
            df['NETRAD'] = SW_IN - SW_OUT + LW_IN - LW_OUT
        else:
            df['NETRAD'] = np.nan

    ta_cols = select_columns(df, 'TA')
    pa_cols = select_columns(df, 'PA')
    p_cols = select_columns(df, 'P')
    le_cols = select_columns(df, 'LE')
    h_cols = select_columns(df, 'H')
    gpp_cols = select_columns(df, 'GPP')
    rh_cols = select_columns(df, 'RH')
    vpd_cols = select_columns(df, 'VPD')
    G_cols = select_columns(df, 'G')
    print(ta_cols, le_cols, NETRAD_cols, G_cols, gpp_cols)
    df['TA_F'] = df[ta_cols].mean(axis=1) if ta_cols else np.nan
    df['PA_F'] = df[pa_cols].mean(axis=1) if pa_cols else np.nan
    df['P_F'] = df[p_cols].mean(axis=1) if p_cols else np.nan
    df['LE_F_MDS'] = df[le_cols].mean(axis=1) if le_cols else np.nan
    df['H_F_MDS'] = df[h_cols].mean(axis=1) if h_cols else np.nan
    df['VPD_F'] = df[vpd_cols].mean(axis=1) / 10 if vpd_cols else np.nan  # Convert VPD from hPa to kPa
    df['RH'] = df[rh_cols].mean(axis=1) if rh_cols else np.nan

    if vpd_cols == [] and rh_cols:
        df['VPD_F'] = calculate_VPD_from_RH(df['TA_F'], df['RH'])
    if rh_cols == [] and vpd_cols:
        df['RH'] = calculate_RH_from_VPD(df['TA_F'], df['VPD_F'])

    mask_vpd = df['VPD_F'].isna() & df['RH'].notna()
    df.loc[mask_vpd, 'VPD_F'] = calculate_VPD_from_RH(df.loc[mask_vpd, 'TA_F'], df.loc[mask_vpd, 'RH'])
    mask_rh = df['RH'].isna() & df['VPD_F'].notna()
    df.loc[mask_rh, 'RH'] = calculate_RH_from_VPD(df.loc[mask_rh, 'TA_F'], df.loc[mask_rh, 'VPD_F'])

    if gpp_cols:
        df['GPP_NT_VUT_REF'] = df[gpp_cols].mean(axis=1)
    else:
        df['GPP_NT_VUT_REF'] = np.nan

    df['G_F_MDS'] = df[G_cols].mean(axis=1) if G_cols else np.nan

    G = df['G_F_MDS'].values
    Rn = df['NETRAD'].values
    LE = df['LE_F_MDS'].values
    H = df['H_F_MDS'].values
    G[np.isnan(G)] = 0
    mask = np.isnan(Rn)
    Rn[mask] = LE[mask] + H[mask]

    LE_CORR_back, H_CORR_back = flux_c(LE, H, Rn, G)
    df['LE_CORR'] = df.get('LE_CORR', pd.Series(LE_CORR_back, index=df.index)).fillna(pd.Series(LE_CORR_back, index=df.index))
    df['H_CORR'] = df.get('H_CORR', pd.Series(H_CORR_back, index=df.index)).fillna(pd.Series(H_CORR_back, index=df.index))

    required_columns = ['LE_F_MDS', 'TA_F', 'P_F', 'LE_CORR', 'H_F_MDS', 'H_CORR',
                        'NETRAD', 'G_F_MDS', 'PA_F', 'VPD_F', 'GPP_NT_VUT_REF', 'WS_F', 'USTAR']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan
    df = pd.concat([df, pd.DataFrame({'theta_1': theta_1, 'theta_2': theta_2})], axis=1)
    df = df[['DateTime', 'LE_F_MDS', 'LE_CORR', 'H_F_MDS', 'H_CORR', 'NETRAD', 'G_F_MDS',
             'TA_F', 'P_F', 'PA_F', 'VPD_F', 'theta_1', 'theta_2', 'GPP_NT_VUT_REF',
             'WS_F', 'USTAR']]
    df.loc[df['PA_F'] > 400, 'PA_F'] /= 1000  # Pa to kPa
    return df
#--------------------------------------------------------------------------------------------
# Read ICOS Data
def read_ICOS(dir_1, parquet_file):
    site_name = parquet_file.split("_")[1]
    df = pd.read_parquet(os.path.join(dir_1, parquet_file))
    df.replace(-9999, np.nan, inplace=True)
    df['DateTime'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")
    swc_cols = [c for c in df.columns if c.startswith("SWC") and not c.endswith("_QC")]
    if len(swc_cols) == 0:
        theta_1 = np.full(len(df), np.nan)
        theta_2 = np.full(len(df), np.nan)
    else:
        theta_1 = df[swc_cols[0]] / 100            # Surface Soil Moisture (m3 m-3)
        theta_2 = df[swc_cols].mean(axis=1) / 100  # Effective Soil Moisture (m3 m-3)
    required_columns = ['LE_F_MDS', 'TA_F', 'P_F', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS',
                        'H_F_MDS_QC', 'H_CORR', 'NETRAD', 'G_F_MDS', 'PA_F', 'VPD_F',
                        'GPP_NT_VUT_REF', 'WS_F', 'USTAR']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Compute OFC fallback if LE_CORR is not provided by the file
    if df['LE_CORR'].isna().all():
        _G  = df['G_F_MDS'].fillna(0).values
        _Rn = df['NETRAD'].values.copy()
        _LE = df['LE_F_MDS'].values
        _H  = df['H_F_MDS'].values
        _mask = np.isnan(_Rn)
        _Rn[_mask] = _LE[_mask] + _H[_mask]
        _lec, _hec = flux_c(_LE, _H, _Rn, _G)
        df['LE_CORR'] = df['LE_CORR'].fillna(pd.Series(_lec, index=df.index))
        df['H_CORR']  = df['H_CORR'].fillna(pd.Series(_hec,  index=df.index))

    df = pd.concat([df, pd.DataFrame({'theta_1': theta_1, 'theta_2': theta_2})], axis=1)
    df = df[['DateTime', 'LE_F_MDS', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS', 'H_F_MDS_QC', 'H_CORR', 'NETRAD', 'G_F_MDS',
             'TA_F', 'P_F', 'PA_F', 'VPD_F', 'theta_1', 'theta_2', 'GPP_NT_VUT_REF',
             'WS_F', 'USTAR']]
    df.loc[df['PA_F'] > 400, 'PA_F'] /= 1000  # kPa
    df['VPD_F'] = df['VPD_F'] / 10  # hPa to kPa
    return df
#--------------------------------------------------------------------------------------------
# Read FLUXNET2015 Data
def read_fluxnet2015(dir_1, zip_file):
    source = zip_file.split("_")[0]
    site_name = zip_file.split("_")[1]
    zip_path = os.path.join(dir_1, zip_file)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            target_files = [
                info for info in zip_ref.infolist()
                if ("FLUXNET2015_FULLSET_HH" in info.filename or "FLUXNET2015_FULLSET_HR" in info.filename)]
            if not target_files:
                print(f"No FLUXNET file found for {site_name} | Source: FLUXNET2015")
                return None
            target_file = target_files[0].filename
            with zip_ref.open(target_file) as f:
                df = pd.read_csv(f)
    except zipfile.BadZipFile:
        print(f"Error: {zip_file} is not a valid zip file.")
        return None
    except FileNotFoundError:
        print(f"Error: {zip_file} not found in the directory.")
        return None
    except Exception as e:
        print(f"Error while processing {zip_file}: {e}")
        return None

    df.replace(-9999, np.nan, inplace=True)

    df['DateTime'] = pd.to_datetime(df['TIMESTAMP_START'], format="%Y%m%d%H%M")

    swc_cols = [c for c in df.columns if c.startswith("SWC") and not c.endswith("_QC")]
    if len(swc_cols) == 0:
        theta_1 = np.full(len(df), np.nan)
        theta_2 = np.full(len(df), np.nan)
    else:
        theta_1 = df[swc_cols[0]] / 100            # Surface Soil Moisture (m3 m-3)
        theta_2 = df[swc_cols].mean(axis=1) / 100  # Effective Soil Moisture (m3 m-3)
    required_columns = ['LE_F_MDS', 'TA_F', 'P_F', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS',
                        'H_F_MDS_QC', 'H_CORR', 'NETRAD', 'G_F_MDS', 'PA_F', 'VPD_F',
                        'GPP_NT_VUT_REF', 'WS_F', 'USTAR']
    for col in required_columns:
        if col not in df.columns:
            df[col] = np.nan

    # Compute OFC fallback if LE_CORR is not provided by the file
    if df['LE_CORR'].isna().all():
        _G  = df['G_F_MDS'].fillna(0).values
        _Rn = df['NETRAD'].values.copy()
        _LE = df['LE_F_MDS'].values
        _H  = df['H_F_MDS'].values
        _mask = np.isnan(_Rn)
        _Rn[_mask] = _LE[_mask] + _H[_mask]
        _lec, _hec = flux_c(_LE, _H, _Rn, _G)
        df['LE_CORR'] = df['LE_CORR'].fillna(pd.Series(_lec, index=df.index))
        df['H_CORR']  = df['H_CORR'].fillna(pd.Series(_hec,  index=df.index))

    df = pd.concat([df, pd.DataFrame({'theta_1': theta_1, 'theta_2': theta_2})], axis=1)
    df = df[['DateTime', 'LE_F_MDS', 'LE_F_MDS_QC', 'LE_CORR', 'H_F_MDS', 'H_F_MDS_QC', 'H_CORR', 'NETRAD', 'G_F_MDS',
             'TA_F', 'P_F', 'PA_F', 'VPD_F', 'theta_1', 'theta_2', 'GPP_NT_VUT_REF',
             'WS_F', 'USTAR']]
    df.loc[df['PA_F'] > 400, 'PA_F'] /= 1000  # Pa to kPa
    df['VPD_F'] = df['VPD_F'] / 10  # hPa to kPa
    return df