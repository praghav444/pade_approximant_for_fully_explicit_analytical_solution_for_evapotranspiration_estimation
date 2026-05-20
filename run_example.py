"""
run_example.py
==============
Demonstrates the four ET models using a single FLUXNET site (US-MMS, Morgan Monroe State Forest, Indiana, USA).

The example input file contains half-hourly observations from US-MMS that
have already passed quality control and preprocessing (as described in the
paper).  The script reads these inputs, applies all four ET models, prints
a summary of errors, and saves the predictions to a CSV file.

Usage
-----
    python run_example.py

Output
------
    results/US-MMS_predictions.csv  : input variables plus model predictions

Required packages
-----------------
    numpy, pandas, scipy
"""

import os
import numpy as np
import pandas as pd
from ET_models import ET_PM, ET_McColl, ET_Pade1, ET_Pade2

INPUT_FILE = os.path.join('example_data', 'US-MMS_example_input.csv')
OUT_DIR    = 'results'
os.makedirs(OUT_DIR, exist_ok=True)


def rmse(pred, obs):
    return np.sqrt(np.mean((pred - obs) ** 2))


def mbe(pred, obs):
    return np.mean(pred - obs)


# Load example data
print(f"Loading {INPUT_FILE} ...")
df = pd.read_csv(INPUT_FILE, parse_dates=['DateTime'])
print(f"  {len(df)} half-hourly observations loaded.")

# Extract input arrays
Rn     = df['Rn'].values
G      = df['G'].values
Ta_C   = df['Ta_C'].values
P_kPa  = df['P_kPa'].values
qa     = df['qa'].values
ga     = df['ga'].values
gs     = df['gs'].values
rho    = df['rho_a'].values
LE_obs = df['LE_obs'].values

# Run all four ET models
kw = dict(P_kPa=P_kPa, rho=rho)
lE_PM              = ET_PM    (Rn, G, Ta_C, qa, ga, gs, **kw)
lE_McColl          = ET_McColl(Rn, G, Ta_C, qa, ga, gs, **kw)
lE_Pade1, x_pade1  = ET_Pade1 (Rn, G, Ta_C, qa, ga, gs, **kw)
lE_Pade2, x_pade2  = ET_Pade2 (Rn, G, Ta_C, qa, ga, gs, **kw)

# Keep only time steps where all four models return a finite value
ok = (np.isfinite(lE_PM) & np.isfinite(lE_McColl) &
      np.isfinite(lE_Pade1) & np.isfinite(lE_Pade2))

print(f"\nSite: US-MMS  |  valid time steps: {ok.sum()} of {len(df)}")
print(f"\n{'Model':<12} {'RMSE (W/m2)':>12} {'MBE (W/m2)':>12}")
print('-' * 38)
for pred, label in [(lE_PM,    'PM'),
                    (lE_McColl, 'McColl'),
                    (lE_Pade1,  'Pade-1'),
                    (lE_Pade2,  'Pade-2')]:
    print(f"  {label:<10} {rmse(pred[ok], LE_obs[ok]):>12.3f} "
          f"{mbe(pred[ok], LE_obs[ok]):>+12.3f}")

# Save results
out = df.copy()
out['lE_PM']     = lE_PM
out['lE_McColl'] = lE_McColl
out['lE_Pade1']  = lE_Pade1
out['lE_Pade2']  = lE_Pade2
out['x_pade1']   = x_pade1
out['x_pade2']   = x_pade2

out_path = os.path.join(OUT_DIR, 'US-MMS_predictions.csv')
out.to_csv(out_path, index=False)
print(f"\nPredictions saved to {out_path}")
