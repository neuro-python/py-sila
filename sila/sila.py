"""
SILA - Sampled Iterative Local Approximation with smoothing kernel optimization
Wrapper around ILLA that optimizes smoothing kernel via cross-validation
"""

import numpy as np
import pandas as pd
from .illa import illa
from .estimate import sila_estimate


def sila(age, value, subid, dt, val0, maxi, sk=None):
    """
    SILA with automatic smoothing kernel optimization

    This function wraps ILLA and optimizes the smoothing kernel by minimizing
    backward prediction residuals. If sk is not provided, it searches from
    0 to 0.5 in steps of 0.05. If sk is provided, it uses those specific
    kernel values.

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
        Maximum iterations for integration
    sk : array-like or None
        Smoothing kernel values to test. If None, use default range [0, 0.05, ..., 0.5]

    Returns:
    --------
    tsila : pd.DataFrame
        Integrated curve from optimal kernel (same format as ILLA tout)
    tdrs : pd.DataFrame
        Discrete rate samples from optimal kernel (same format as ILLA tdrs)

    Reference:
    ----------
    Betthauser, T.J., et al. (2022). Brain, 145(11), 4059-4071.
    """
    # Convert inputs to numpy arrays
    age = np.asarray(age, dtype=np.float64)
    value = np.asarray(value, dtype=np.float64)
    subid = np.asarray(subid)

    # Create table for preprocessing
    t = pd.DataFrame({
        'age': age,
        'val': value,
        'subid': subid
    })
    t = t.sort_values(['subid', 'age']).reset_index(drop=True)

    # Get unique subjects
    subs = np.unique(t['subid'])

    # Create observation index and number of scans per subject
    t['idx'] = 0
    t['ns'] = 0

    for sub in subs:
        ids = t['subid'] == sub
        ages = t.loc[ids, 'age'].values
        sort_idx = np.argsort(ages)

        # Assign ordered observation numbers (1-indexed like MATLAB)
        t.loc[ids, 'idx'] = sort_idx + 1
        t.loc[ids, 'ns'] = len(ages)

    # Remove cases without longitudinal data
    t = t[t['ns'] > 1].reset_index(drop=True)

    # Calculate ratio for weighting residuals (A+ vs A-)
    # This balances contribution from positive and negative cases
    last_obs = t[t['idx'] == t['ns']]
    n_pos = np.sum(last_obs['val'] >= val0)
    n_neg = np.sum(last_obs['val'] < val0)

    if n_neg == 0:
        resnorm = 1.0
    else:
        resnorm = n_pos / n_neg

    # Indices for biomarker positive and negative cases
    idpos = t['val'] > val0
    idneg = t['val'] <= val0

    # Determine smoothing kernels to test
    if sk is None:
        # Default: optimize from 0 to 0.5 in steps of 0.05
        sk = np.arange(0, 0.55, 0.05)
    else:
        sk = np.asarray(sk, dtype=np.float64)
        if sk.ndim == 0:
            sk = np.array([sk])

    # Storage for results from each kernel
    dat_tilla = []
    dat_tdrs = []
    SSQpos = np.zeros(len(sk))
    SSQneg = np.zeros(len(sk))

    # Test each smoothing kernel
    for i, kernel in enumerate(sk):
        # Run ILLA with current kernel
        tilla, tdrs = illa(t['age'].values, t['val'].values, t['subid'].values,
                           dt, val0, maxi, kernel)

        dat_tilla.append(tilla)
        dat_tdrs.append(tdrs)

        # Estimate values using backwards prediction (align to last observation)
        temp = sila_estimate(tilla, t['age'].values, t['val'].values,
                             t['subid'].values,
                             align_event='last', truncate_aget0='no')

        # Calculate sum of squared residuals for positive and negative cases
        SSQpos[i] = np.sum(temp.loc[idpos, 'estresid'] ** 2)
        SSQneg[i] = np.sum(temp.loc[idneg, 'estresid'] ** 2)

    # Find optimal kernel (minimize weighted sum of squared residuals)
    weighted_ssq = SSQpos + resnorm * SSQneg
    ids = np.argmin(weighted_ssq)

    # Return results from optimal kernel
    tsila = dat_tilla[ids].copy()
    tdrs = dat_tdrs[ids].copy()

    # Add optimal smoothing kernel to output
    tdrs['skern'] = sk[ids]

    return tsila, tdrs
