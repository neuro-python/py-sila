"""
ILLA - Iterative Local Linear Approximation
Core implementation with exact MATLAB numerical equivalence
"""

import numpy as np
import pandas as pd
from .utils import rloess_smooth, polyfit_matlab


def illa(age, value, subid, dt, val0, maxi, skern):
    """
    Iterative Local Linear Approximation using discrete rate sampling and Euler integration

    This function estimates the biomarker vs. time curve from longitudinal data using:
    1. Within-subject linear slopes (polyfit)
    2. Discrete rate sampling at biomarker values
    3. LOESS smoothing of rate vs. value curve (optional)
    4. Forward and backward Euler integration from starting point

    Parameters:
    -----------
    age : array-like
        Age at observation
    value : array-like
        Observed biomarker values
    subid : array-like
        Subject identifiers
    dt : float
        Integration step size (e.g., 0.25 years)
    val0 : float
        Threshold value where time = 0 (e.g., 0.79 for amyloid)
    maxi : int
        Maximum iterations for forward/backward integration
    skern : float
        Smoothing kernel span (0-1). If 0, no smoothing applied.

    Returns:
    --------
    tout : pd.DataFrame
        Integrated curve with columns:
        - val: biomarker values along curve
        - time: time relative to curve start
        - adtime: time relative to val0 threshold
        - mrate: mean rate at each point
        - sdrate: standard deviation of rate
        - nsubs: number of subjects contributing to rate estimate
        - sdval: error estimate in biomarker value
        - ci95: 95% confidence interval

    tdrs : pd.DataFrame
        Discrete rate samples with columns:
        - val: biomarker value
        - rate: estimated rate of change (weighted mean)
        - ratestd: standard deviation of rates
        - npos: number of positive rates
        - tot: total number of subjects
        - ci: 95% confidence interval of rate
        - skern: smoothing kernel used

    Reference:
    ----------
    Betthauser, T.J., et al. (2022). Brain, 145(11), 4059-4071.
    """
    # Convert inputs to numpy arrays with float64 precision
    age = np.asarray(age, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    subid = np.asarray(subid)

    # Suppress warnings (matching MATLAB behavior)
    import warnings
    warnings.filterwarnings('ignore')

    # Get unique subjects
    subs = np.unique(subid)

    # Create table object (DataFrame)
    t = pd.DataFrame({
        'age': age,
        'val': value,
        'subid': subid
    })

    # Sort by subject ID then age
    t = t.sort_values(['subid', 'age']).reset_index(drop=True)

    # Initialize columns for within-person statistics
    t['mrate'] = np.nan
    t['max'] = np.nan
    t['min'] = np.nan
    t['count'] = np.nan
    t['nvis'] = np.nan

    # Calculate within-person slopes using polyfit
    for i, sub in enumerate(subs):
        idsub = t['subid'] == sub
        tsub = t.loc[idsub].copy()

        if len(tsub) > 1:
            # Fit linear model: value = p[0]*age + p[1]
            p = polyfit_matlab(tsub['age'].values, tsub['val'].values, 1)
            t.loc[idsub, 'mrate'] = p[0]  # slope
            t.loc[idsub, 'max'] = tsub['val'].max()
            t.loc[idsub, 'min'] = tsub['val'].min()
            # Create ordered observation numbers (1-indexed like MATLAB)
            t.loc[idsub, 'count'] = np.arange(1, len(tsub) + 1)

        t.loc[idsub, 'nvis'] = len(tsub)

    # Remove cases with only one scan
    tmod = t[t['nvis'] > 1].copy()

    # Create discrete query values (qval)
    val_range = tmod['val'].max() - tmod['val'].min()
    qval = np.arange(tmod['val'].min(), tmod['val'].max() + val_range/150, val_range/150)

    # Keep only first observation per subject
    tmod = tmod[tmod['count'] == 1].copy()

    # Remove unrealistic rates (>100)
    tmod = tmod[tmod['mrate'] < 100].copy()

    # Perform discrete rate sampling
    tdrs_data = []
    rate_list = []
    vals_list = []

    for qv in qval:
        # Find subjects whose range includes this query value
        ids = (tmod['min'] < qv) & (tmod['max'] > qv)

        if np.sum(ids) == 0:
            continue

        # Weighted average of rates (weighted by number of visits)
        rates_subset = tmod.loc[ids, 'mrate'].values
        nvis_subset = tmod.loc[ids, 'nvis'].values

        rate_weighted = np.sum(rates_subset * nvis_subset) / np.sum(nvis_subset)
        rate_std = np.std(rates_subset, ddof=1)  # Use ddof=1 to match MATLAB std
        npos = np.sum(rates_subset > 0)
        tot = np.sum(ids)

        tdrs_data.append({
            'val': qv,
            'rate': rate_weighted,
            'ratestd': rate_std,
            'npos': npos,
            'tot': tot
        })

        # Store for smoothing
        rate_list.extend(rates_subset.tolist())
        vals_list.extend([qv] * len(rates_subset))

    tdrs = pd.DataFrame(tdrs_data)

    # Calculate 95% CI
    tdrs['ci'] = 1.96 * tdrs['ratestd'] / np.sqrt(tdrs['tot'])

    # Keep only observations with at least 2 subjects
    tdrs = tdrs[tdrs['tot'] >= 2].reset_index(drop=True)

    # Apply robust LOESS smoothing if skern != 0
    if skern != 0:
        rate_array = np.array(rate_list, dtype=np.float64)
        vals_array = np.array(vals_list, dtype=np.float64)

        srate = rloess_smooth(vals_array, rate_array, skern, robust=True)

        # Get unique values and their indices
        vals_unique, unique_idx = np.unique(vals_array, return_index=True)
        srate_unique = srate[unique_idx]

        # Map smoothed rates back to tdrs values
        smoothed_rates = []
        for val in tdrs['val']:
            idx = np.where(vals_unique == val)[0]
            if len(idx) > 0:
                smoothed_rates.append(srate_unique[idx[0]])
            else:
                # If exact match not found, use nearest
                nearest_idx = np.argmin(np.abs(vals_unique - val))
                smoothed_rates.append(srate_unique[nearest_idx])

        tdrs['rate'] = smoothed_rates

    tdrs['skern'] = skern

    # Determine if curve is increasing or decreasing
    med_rate = np.median(tdrs['rate'])

    # Forward integration (Euler's method)
    qval_cur = np.mean(tdrs['val'])
    valf = []
    rf = []
    sdf = []
    nf = []
    nif = 0

    while qval_cur < tdrs['val'].max() and nif < maxi:
        # Check stopping condition for decreasing curves
        if med_rate < 0 and qval_cur < tdrs['val'].min():
            break

        # Find closest discrete rate value
        idx = np.argmin(np.abs(tdrs['val'] - qval_cur))

        # Check stopping conditions
        if tdrs.iloc[idx]['tot'] < 2:
            break
        if tdrs.iloc[idx]['rate'] <= 0 and med_rate > 0:
            break
        if tdrs.iloc[idx]['rate'] >= 0 and med_rate < 0:
            break

        # Store current values
        valf.append(qval_cur)
        rf.append(tdrs.iloc[idx]['rate'])
        sdf.append(tdrs.iloc[idx]['ratestd'])
        nf.append(tdrs.iloc[idx]['tot'])

        # Euler's method: next value = current + rate * dt
        qval_cur = tdrs.iloc[idx]['rate'] * dt + qval_cur
        nif += 1

    tf = np.cumsum(dt * np.ones(nif)) - dt

    # Backward integration (Euler's method)
    qval_cur = np.mean(tdrs['val'])
    valb = []
    rb = []
    sdb = []
    nb = []
    nib = 0

    while qval_cur > qval.min() and nib < maxi:
        # Check stopping condition for decreasing curves
        if med_rate < 0 and qval_cur > tdrs['val'].max():
            break

        # Find closest discrete rate value
        idx = np.argmin(np.abs(tdrs['val'] - qval_cur))

        # Check stopping conditions
        if tdrs.iloc[idx]['tot'] < 2:
            break
        if tdrs.iloc[idx]['rate'] < 0 and med_rate > 0:
            break
        if tdrs.iloc[idx]['rate'] > 0 and med_rate < 0:
            break

        # Store current values
        valb.append(qval_cur)
        rb.append(tdrs.iloc[idx]['rate'])
        sdb.append(tdrs.iloc[idx]['ratestd'])
        nb.append(tdrs.iloc[idx]['tot'])

        # Backward step: subtract rate * dt
        qval_cur = tdrs.iloc[idx]['rate'] * (-dt) + qval_cur
        nib += 1

    tb = -np.cumsum(dt * np.ones(nib)) + dt

    # Create output table by combining forward and backward
    # Note: MATLAB does flip(arr(2:end)) which means skip first element THEN flip
    # This is equivalent to Python arr[1:][::-1] or np.flip(arr[1:])
    tout = pd.DataFrame({
        'val': np.concatenate([np.flip(valb[1:]), valf]),
        'time': np.concatenate([np.flip(tb[1:]), tf]),
        'mrate': np.concatenate([np.flip(rb[1:]), rf]),
        'sdrate': np.concatenate([np.flip(sdb[1:]), sdf]),
        'nsubs': np.concatenate([np.flip(nb[1:]), nf])
    })

    # Handle edge case of empty output
    if len(tout) == 0:
        # Return empty DataFrames with proper structure
        tout['adtime'] = []
        tout['sdval'] = []
        tout['ci95'] = []
        return tout, tdrs

    # Calculate adtime (time relative to val0 threshold)
    id0 = np.argmin(np.abs(tout['val'] - val0))
    tout['adtime'] = tout['time'] - tout['time'].iloc[id0]

    # Error propagation for model uncertainty
    # Type I error propagation
    rate_error_squared = (tout['sdrate'] ** 2) / (tout['mrate'] ** 2)
    rate_contribution = ((tout['mrate'] * dt) * np.sqrt(rate_error_squared)) ** 2
    pet_error = (tout['val'] * 0.05) ** 2  # Approximating 5% error in PET

    tout['sdval'] = np.sqrt(rate_contribution + pet_error)
    tout['ci95'] = 1.96 * tout['sdval'] / np.sqrt(tout['nsubs'])

    return tout, tdrs
