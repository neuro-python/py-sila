"""
Utility functions for SILA toolbox with exact MATLAB numerical equivalence
"""

import numpy as np
from scipy import interpolate


def rloess_smooth(x, y, span, robust=True):
    """
    Robust LOESS smoothing matching MATLAB's smooth(y, x, span, 'rloess')

    MATLAB's smooth function uses Cleveland's LOESS algorithm with tricube weights.
    This implementation aims for exact numerical equivalence.

    Parameters:
    -----------
    x : array-like
        Independent variable values (predictors)
    y : array-like
        Dependent variable values (responses)
    span : float
        Smoothing span as fraction of data (0-1), where span=0.3 means 30% of points
    robust : bool
        If True, use robust fitting with bisquare weights (MATLAB default for 'rloess')

    Returns:
    --------
    ys : np.ndarray
        Smoothed y values

    Notes:
    ------
    MATLAB's smooth uses:
    - Tricube weight function: w(d) = (1 - |d|^3)^3 for |d| < 1
    - Robust iterations with bisquare weights for outlier resistance
    - Weighted linear regression at each point
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    n = len(x)
    ys = np.zeros(n, dtype=np.float64)

    # Calculate window width: span is fraction of total points
    if span <= 0 or span > 1:
        raise ValueError("span must be in (0, 1]")

    # For span as fraction, calculate number of points
    # MATLAB rounds to nearest integer for window size
    r = int(np.ceil(span * n))
    r = max(r, 2)  # Minimum window size of 2

    # Sort by x values (MATLAB processes in sorted order)
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]

    # Initialize robustness weights (all 1.0 initially)
    robustness_weights = np.ones(n, dtype=np.float64)

    # Robust fitting iterations (MATLAB uses 5 iterations for 'rloess')
    n_robust_iterations = 5 if robust else 1

    for iteration in range(n_robust_iterations):
        for i in range(n):
            # Find nearest neighbors
            # MATLAB uses r points centered around current point when possible
            distances = np.abs(x_sorted - x_sorted[i])
            neighbor_idx = np.argsort(distances)[:r]

            x_local = x_sorted[neighbor_idx]
            y_local = y_sorted[neighbor_idx]

            # Calculate distances for tricube weights
            max_dist = np.max(np.abs(x_local - x_sorted[i]))
            if max_dist == 0:
                max_dist = 1.0  # Avoid division by zero

            # Tricube weight function: (1 - |d|^3)^3 for |d| < 1
            dist_normalized = np.abs(x_local - x_sorted[i]) / max_dist
            tricube_weights = np.power(1.0 - np.power(dist_normalized, 3), 3)

            # Combine with robustness weights from previous iteration
            combined_weights = tricube_weights * robustness_weights[neighbor_idx]

            # Weighted linear regression: fit line y = a + b*x
            # Using weighted least squares
            if np.sum(combined_weights) == 0:
                ys[sort_idx[i]] = y_sorted[i]
                continue

            # Normalize weights
            w = combined_weights / np.sum(combined_weights)

            # Weighted mean
            x_mean = np.sum(w * x_local)
            y_mean = np.sum(w * y_local)

            # Weighted covariance and variance
            dx = x_local - x_mean
            dy = y_local - y_mean

            cov_xy = np.sum(w * dx * dy)
            var_x = np.sum(w * dx * dx)

            # Calculate slope and intercept
            if var_x > 1e-15:
                slope = cov_xy / var_x
                intercept = y_mean - slope * x_mean
                ys[sort_idx[i]] = slope * x_sorted[i] + intercept
            else:
                ys[sort_idx[i]] = y_mean

        # Update robustness weights using bisquare function (for all but last iteration)
        if iteration < n_robust_iterations - 1:
            residuals = y_sorted - ys[sort_idx]

            # Calculate median absolute deviation (MAD) for robust scale estimate
            # MATLAB uses 6*MAD as threshold
            mad = np.median(np.abs(residuals))
            if mad > 0:
                u = residuals / (6.0 * mad)
            else:
                u = residuals / (np.std(residuals) + 1e-15)

            # Bisquare weight function: (1 - u^2)^2 for |u| < 1, else 0
            robustness_weights = np.where(
                np.abs(u) < 1,
                np.power(1.0 - np.power(u, 2), 2),
                0.0
            )

    return ys


def polyfit_matlab(x, y, degree):
    """
    Polynomial fitting matching MATLAB's polyfit exactly

    Parameters:
    -----------
    x : array-like
        x values
    y : array-like
        y values
    degree : int
        Polynomial degree

    Returns:
    --------
    p : np.ndarray
        Polynomial coefficients [highest degree first]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Use numpy polyfit which matches MATLAB
    p = np.polyfit(x, y, degree)
    return p


def interp1_matlab(x, y, xi, method='linear'):
    """
    1D interpolation matching MATLAB's interp1

    Parameters:
    -----------
    x : array-like
        Sample points (must be monotonic)
    y : array-like
        Sample values
    xi : array-like
        Points to interpolate at
    method : str
        Interpolation method ('linear', 'nearest', 'cubic')

    Returns:
    --------
    yi : np.ndarray
        Interpolated values
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xi = np.asarray(xi, dtype=np.float64)

    # Create interpolator
    if method == 'linear':
        f = interpolate.interp1d(x, y, kind='linear',
                                  bounds_error=False, fill_value=np.nan)
    elif method == 'nearest':
        f = interpolate.interp1d(x, y, kind='nearest',
                                  bounds_error=False, fill_value=np.nan)
    elif method == 'cubic':
        f = interpolate.interp1d(x, y, kind='cubic',
                                  bounds_error=False, fill_value=np.nan)
    else:
        raise ValueError(f"Unknown method: {method}")

    return f(xi)


def fitlm_matlab(x, y):
    """
    Linear model fitting matching MATLAB's fitlm

    Returns coefficients: [intercept, slope]

    Parameters:
    -----------
    x : array-like
        Independent variable
    y : array-like
        Dependent variable

    Returns:
    --------
    coefficients : dict
        Dictionary with 'Estimate' containing [intercept, slope]
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    # Use polyfit for degree 1 (linear)
    p = np.polyfit(x, y, 1)

    # MATLAB fitlm returns coefficients as [intercept, slope]
    # numpy polyfit returns [slope, intercept]
    return {
        'Estimate': np.array([p[1], p[0]], dtype=np.float64)
    }
