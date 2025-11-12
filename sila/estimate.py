"""
SILA_estimate - Subject-level predictions and time-to-threshold estimates
"""

import numpy as np
import pandas as pd
from .utils import interp1_matlab, fitlm_matlab


def sila_estimate(tsila, age, val, subid,
                  align_event='last',
                  extrap_years=3,
                  truncate_aget0='yes'):
    """
    Generate subject-level estimates from SILA model

    This function aligns individual subject data to the population curve and
    estimates:
    - Predicted biomarker values
    - Age at threshold crossing (estaget0)
    - Time from threshold (estdtt0)
    - Residuals from model

    Parameters:
    -----------
    tsila : pd.DataFrame
        Output from SILA or ILLA (integrated curve)
    age : array-like
        Ages at observations
    val : array-like
        Observed biomarker values
    subid : array-like
        Subject identifiers
    align_event : str
        Which observation to use for alignment: 'first', 'last', or 'all'
        Default: 'last'
    extrap_years : float
        Number of years for linear extrapolation beyond model range
        Default: 3
    truncate_aget0 : str
        Whether to truncate estaget0 below minimum model time ('yes' or 'no')
        Default: 'yes'

    Returns:
    --------
    tout : pd.DataFrame
        Subject-level estimates with columns:
        - subid: Subject identifier
        - age: Age at observation
        - val: Observed value
        - minage: Minimum age for subject
        - maxage: Maximum age for subject
        - valt0: Model value at threshold (time=0)
        - ageref: Reference age used for alignment
        - dtageref: Time from reference age
        - estval: Estimated value from model
        - estaget0: Estimated age at threshold crossing
        - estdtt0: Estimated time from threshold
        - estresid: Residual (observed - estimated)
        - estpos: Whether estimated value is above threshold
        - aevent: Alignment event used
        - extrapyrs: Extrapolation years used
        - truncated: Whether estaget0 was truncated (0 or 1)

    Reference:
    ----------
    Betthauser, T.J., et al. (2022). Brain, 145(11), 4059-4071.
    """
    # Convert inputs to appropriate types
    age = np.asarray(age, dtype=np.float64)
    val = np.asarray(val, dtype=np.float64)
    subid = np.asarray(subid)

    # Create extrapolated model
    # Fit linear models to upper and lower portions of curve
    upper_mask = tsila['adtime'] > (tsila['adtime'].max() - extrap_years)
    md1 = fitlm_matlab(tsila.loc[upper_mask, 'adtime'].values,
                       tsila.loc[upper_mask, 'val'].values)

    lower_mask = tsila['adtime'] < (tsila['adtime'].min() + extrap_years)
    md2 = fitlm_matlab(tsila.loc[lower_mask, 'adtime'].values,
                       tsila.loc[lower_mask, 'val'].values)

    # Extract slopes and intercepts
    slopeu = md1['Estimate'][1]  # Upper slope
    intu = md1['Estimate'][0]     # Upper intercept
    slopel = md2['Estimate'][1]   # Lower slope
    intl = md2['Estimate'][0]     # Lower intercept

    mll = tsila['val'].min()
    mul = tsila['val'].max()

    # Resample curve to finer grid (0.01 year spacing)
    tt = np.arange(tsila['adtime'].min(), tsila['adtime'].max() + 0.01, 0.01)
    mval = interp1_matlab(tsila['adtime'].values, tsila['val'].values, tt, method='linear')

    # Create extrapolated values if using 'all' alignment
    if align_event == 'all':
        if tsila['val'].iloc[-1] > tsila['val'].iloc[0]:
            # Increasing with time
            ttl = np.arange(tt.min() * 3, tt.min(), 0.01)
            ttl = ttl[:-1]  # Remove last element
            vall = (ttl - tt.min()) * slopel + mval[0]

            ttu = np.arange(tt.max(), tt.max() * 2 + 0.01, 0.01)
            ttu = ttu[1:]  # Remove first element
            valu = (ttu - tt.max()) * slopeu + mval[-1]

            tt = np.concatenate([ttl, tt, ttu]).astype(np.float32)
            mval = np.concatenate([vall, mval, valu]).astype(np.float32)
        else:
            # Decreasing with time
            ttl = np.arange(tt.min() * 3, tt.min(), 0.01)
            ttl = ttl[:-1]
            vall = (ttl - tt.min()) * slopeu + mval[-1]

            ttu = np.arange(tt.max(), tt.max() * 2 + 0.01, 0.01)
            ttu = ttu[1:]
            valu = (ttu - tt.max()) * slopel + mval[0]

            tt = np.concatenate([ttl, tt, ttu]).astype(np.float32)
            mval = np.concatenate([vall, mval, valu]).astype(np.float32)

    # Get value at threshold (time = 0)
    valt0_idx = np.argmin(np.abs(tt))
    valt0 = mval[valt0_idx]

    # Create output table
    tout = pd.DataFrame({
        'subid': subid,
        'age': age,
        'val': val
    })
    tout = tout.sort_values(['subid', 'age']).reset_index(drop=True)

    # Initialize output columns
    tout['minage'] = np.nan
    tout['maxage'] = np.nan
    tout['valt0'] = valt0
    tout['ageref'] = np.nan
    tout['dtageref'] = np.nan
    tout['estval'] = np.nan
    tout['estaget0'] = np.nan
    tout['estdtt0'] = np.nan
    tout['estresid'] = np.nan

    # Process each observation
    for i in range(len(tout)):
        # Get all observations for this subject
        ids = tout['subid'] == tout.iloc[i]['subid']
        tsub = tout.loc[ids].copy()

        tout.loc[i, 'minage'] = tsub['age'].min()
        tout.loc[i, 'maxage'] = tsub['age'].max()

        if align_event == 'all':
            if len(tsub) == 1:
                # Single scan: place on curve
                id0 = np.argmin(np.abs(mval - tout.iloc[i]['val']))
                tout.loc[i, 'ageref'] = tsub['age'].iloc[0]
                tout.loc[i, 'dtageref'] = 0
                tout.loc[i, 'estval'] = mval[id0]
                tout.loc[i, 'estaget0'] = tout.iloc[i]['age'] - tt[id0]
                tout.loc[i, 'estdtt0'] = tout.iloc[i]['age'] - (tout.iloc[i]['age'] - tt[id0])
            else:
                # Multiple scans: minimize SSQ across all scans
                idmove = np.round((tsub['age'].values - tsub['age'].iloc[0]) / 0.01).astype(int)
                ll = int(np.max(idmove + 1))  # Convert to scalar
                ul = int(len(tt) - idmove.max())  # Convert to scalar

                # Generate all possible query indices
                if ll >= ul:
                    # Not enough range, fall back to single point alignment
                    id0 = np.argmin(np.abs(mval - tout.iloc[i]['val']))
                    tout.loc[i, 'ageref'] = tsub['age'].iloc[0]
                    tout.loc[i, 'dtageref'] = 0
                    tout.loc[i, 'estval'] = mval[id0]
                    tout.loc[i, 'estaget0'] = tout.iloc[i]['age'] - tt[id0]
                    tout.loc[i, 'estdtt0'] = tout.iloc[i]['age'] - (tout.iloc[i]['age'] - tt[id0])
                    continue

                idts = np.add.outer(np.arange(ll, ul), idmove)
                smval = mval[idts]

                # Find set that minimizes sum of squared residuals
                # Ensure proper broadcasting: smval is (n_positions, n_scans)
                # tsub['val'] should be (n_scans,) -> reshape to (n_scans, 1) for broadcasting
                tsub_vals_col = tsub['val'].values.reshape(-1, 1)
                ssq = np.sum((smval.T - tsub_vals_col) ** 2, axis=0)
                id_opt = np.argmin(ssq)
                id_optim = idts[id_opt, 0]

                iddt = int(np.round((tout.iloc[i]['age'] - tsub['age'].iloc[0]) / 0.01))
                idt = id_optim + iddt

                # Ensure idt is within bounds
                idt = max(0, min(idt, len(mval) - 1))

                tout.loc[i, 'ageref'] = tsub['age'].mean()
                tout.loc[i, 'dtageref'] = tout.iloc[i]['age'] - tsub['age'].mean()
                tout.loc[i, 'estval'] = mval[idt]
                tout.loc[i, 'estaget0'] = tsub['age'].iloc[0] - tt[id_optim]
                tout.loc[i, 'estdtt0'] = tout.iloc[i]['age'] - (tsub['age'].iloc[0] - tt[id_optim])

        elif align_event == 'first':
            # Align to first observation
            id0 = np.argmin(np.abs(mval - tsub['val'].iloc[0]))
            tout.loc[i, 'ageref'] = tsub['age'].iloc[0]
            tout.loc[i, 'dtageref'] = tout.iloc[i]['age'] - tsub['age'].iloc[0]

            dt_idx = int(np.round((tout.iloc[i]['age'] - tsub['age'].iloc[0]) / 0.01))

            # Check if extrapolation needed
            # MATLAB: id0>=numel(mval) || id0 + dt_idx > numel(mval)
            # Python needs to check both index bounds AND value bounds
            if (id0 >= len(mval) - 1 and tsub['val'].iloc[0] > mval[-1]) or id0 + dt_idx > len(mval) - 1:
                # Extrapolate at the top (upper bound)
                chronfirst = (tsub['val'].iloc[0] - intu) / slopeu
                tout.loc[i, 'estval'] = (chronfirst + tout.iloc[i]['dtageref']) * slopeu + intu
                tout.loc[i, 'estdtt0'] = chronfirst + tout.iloc[i]['dtageref']
                tout.loc[i, 'estaget0'] = tsub['age'].iloc[0] - chronfirst
            elif id0 == 0 and tsub['val'].iloc[0] < mval[0]:
                # Extrapolate from the bottom (lower bound)
                chronfirst = (tsub['val'].iloc[0] - intl) / slopel
                tout.loc[i, 'estval'] = (chronfirst + tout.iloc[i]['dtageref']) * slopel + intl
                tout.loc[i, 'estdtt0'] = chronfirst + tout.iloc[i]['dtageref']
                tout.loc[i, 'estaget0'] = tsub['age'].iloc[0] - chronfirst
            else:
                idx_lookup = id0 + dt_idx
                if idx_lookup < 0:
                    idx_lookup = 0
                elif idx_lookup >= len(mval):
                    idx_lookup = len(mval) - 1
                tout.loc[i, 'estval'] = mval[idx_lookup]
                tout.loc[i, 'estaget0'] = tsub['age'].iloc[0] - tt[id0]
                tout.loc[i, 'estdtt0'] = tout.iloc[i]['age'] - (tsub['age'].iloc[0] - tt[id0])

        elif align_event == 'last':
            # Align to last observation
            id0 = np.argmin(np.abs(mval - tsub['val'].iloc[-1]))
            tout.loc[i, 'ageref'] = tsub['age'].iloc[-1]
            tout.loc[i, 'dtageref'] = tout.iloc[i]['age'] - tsub['age'].iloc[-1]

            dt_idx = int(np.round((tout.iloc[i]['age'] - tsub['age'].iloc[-1]) / 0.01))

            # Check if extrapolation needed
            # MATLAB checks: id0 + dt_idx < 1 (lower bound) or id0 >= numel(mval) (upper bound)
            # For upper extrapolation, we need to check if value exceeds curve range
            # since argmin will just return the last index even if value is beyond curve
            if id0 + dt_idx < 0 or id0 == 0:
                # Lower extrapolation
                chronend = (tsub['val'].iloc[-1] - intl) / slopel
                tout.loc[i, 'estval'] = (chronend + tout.iloc[i]['dtageref']) * slopel + intl
                tout.loc[i, 'estdtt0'] = chronend + tout.iloc[i]['dtageref']
                tout.loc[i, 'estaget0'] = tsub['age'].iloc[-1] - chronend
            elif id0 >= len(mval) - 1 and tsub['val'].iloc[-1] > mval[-1]:
                # Upper extrapolation: last obs value exceeds curve maximum
                chronend = (tsub['val'].iloc[-1] - intu) / slopeu
                tout.loc[i, 'estval'] = (chronend + tout.iloc[i]['dtageref']) * slopeu + intu
                tout.loc[i, 'estdtt0'] = chronend + tout.iloc[i]['dtageref']
                tout.loc[i, 'estaget0'] = tsub['age'].iloc[-1] - chronend
            else:
                idx_lookup = id0 + dt_idx
                if idx_lookup < 0:
                    idx_lookup = 0
                elif idx_lookup >= len(mval):
                    idx_lookup = len(mval) - 1
                tout.loc[i, 'estval'] = mval[idx_lookup]
                tout.loc[i, 'estaget0'] = tsub['age'].iloc[-1] - tt[id0]
                tout.loc[i, 'estdtt0'] = tout.iloc[i]['age'] - (tsub['age'].iloc[-1] - tt[id0])

    # Calculate residuals
    tout['estresid'] = tout['val'] - tout['estval']
    tout['estpos'] = tout['estval'] >= valt0

    # Add metadata
    tout['aevent'] = align_event
    tout['extrapyrs'] = extrap_years

    # Truncate estaget0 if requested
    if 'y' in truncate_aget0.lower():
        tout['truncated'] = 0
        for i in range(len(tout)):
            tsub = tout[tout['subid'] == tout.iloc[i]['subid']]
            dtt0_maxage = tsub.loc[tsub['age'] == tsub['maxage'], 'estdtt0'].values

            if len(dtt0_maxage) > 0 and dtt0_maxage[0] < tsila['adtime'].min():
                dtshift = tsila['adtime'].min() - dtt0_maxage[0]
                tout.loc[i, 'estaget0'] = tout.iloc[i]['estaget0'] - dtshift
                tout.loc[i, 'estdtt0'] = tout.iloc[i]['estdtt0'] + dtshift
                tout.loc[i, 'truncated'] = 1
    else:
        tout['truncated'] = 0

    return tout
