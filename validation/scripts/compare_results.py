"""
Validation script to compare MATLAB and Python SILA implementations

This script loads MATLAB reference outputs and compares them against
Python implementation to verify numerical equivalence within 1e-10 tolerance.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sila import sila, sila_estimate


def validate_results(matlab_df, python_df, table_name, tolerance=1e-10):
    """
    Compare MATLAB and Python DataFrames and calculate error metrics

    Parameters:
    -----------
    matlab_df : pd.DataFrame
        MATLAB reference results
    python_df : pd.DataFrame
        Python implementation results
    table_name : str
        Name of table being compared (for reporting)
    tolerance : float
        Maximum acceptable relative error

    Returns:
    --------
    results : dict
        Dictionary containing error metrics and pass/fail status
    """
    print(f"\n{'='*70}")
    print(f"Validating {table_name}")
    print(f"{'='*70}")

    results = {
        'table': table_name,
        'pass': True,
        'max_abs_error': 0,
        'max_rel_error': 0,
        'rmse': 0,
        'correlation': 1.0,
        'failed_columns': []
    }

    # Check dimensions
    if matlab_df.shape != python_df.shape:
        print(f"WARNING: Shape mismatch!")
        print(f"  MATLAB: {matlab_df.shape}")
        print(f"  Python: {python_df.shape}")
        results['pass'] = False
        return results

    print(f"Shape: {matlab_df.shape}")

    # Compare numeric columns
    numeric_cols = matlab_df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col not in python_df.columns:
            print(f"Column '{col}' missing in Python output!")
            results['failed_columns'].append(col)
            results['pass'] = False
            continue

        matlab_vals = matlab_df[col].values
        python_vals = python_df[col].values

        # Skip if all NaN
        if np.all(np.isnan(matlab_vals)) and np.all(np.isnan(python_vals)):
            continue

        # Handle NaN differences
        matlab_nan = np.isnan(matlab_vals)
        python_nan = np.isnan(python_vals)

        if not np.array_equal(matlab_nan, python_nan):
            print(f"  {col}: NaN pattern mismatch!")
            results['failed_columns'].append(col)
            results['pass'] = False
            continue

        # Compare non-NaN values
        valid_mask = ~matlab_nan
        if not np.any(valid_mask):
            continue

        matlab_valid = matlab_vals[valid_mask]
        python_valid = python_vals[valid_mask]

        # Absolute error
        abs_error = np.abs(matlab_valid - python_valid)
        max_abs = np.max(abs_error)

        # Relative error (avoid division by zero)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_error = np.abs(matlab_valid - python_valid) / np.abs(matlab_valid)
            rel_error = rel_error[np.isfinite(rel_error)]

            if len(rel_error) > 0:
                max_rel = np.max(rel_error)
            else:
                max_rel = 0

        # RMSE
        rmse = np.sqrt(np.mean(abs_error ** 2))

        # Correlation
        if len(matlab_valid) > 1 and np.std(matlab_valid) > 1e-15:
            corr = np.corrcoef(matlab_valid, python_valid)[0, 1]
        else:
            corr = 1.0

        # Update maximum errors
        results['max_abs_error'] = max(results['max_abs_error'], max_abs)
        results['max_rel_error'] = max(results['max_rel_error'], max_rel)
        results['rmse'] = max(results['rmse'], rmse)
        results['correlation'] = min(results['correlation'], corr)

        # Check tolerance
        passed = max_rel < tolerance or max_abs < tolerance * 10

        status = "PASS" if passed else "FAIL"
        print(f"  {col:15s}: max_abs={max_abs:.2e}, max_rel={max_rel:.2e}, "
              f"rmse={rmse:.2e}, corr={corr:.6f} [{status}]")

        if not passed:
            results['failed_columns'].append(col)
            results['pass'] = False

    return results


def main():
    """Main validation workflow"""
    print("\n" + "="*70)
    print("SILA MATLAB vs Python Validation")
    print("="*70)

    # Paths
    data_dir = Path(__file__).parent.parent.parent / 'data'
    validation_dir = Path(__file__).parent

    # Load input data
    print("\nLoading input data...")
    try:
        data = pd.read_csv(data_dir / 'sila_input_amyloid.csv')
        print(f"  Data shape: {data.shape}")
        print(f"  Columns: {list(data.columns)}")
    except FileNotFoundError:
        print("ERROR: Input data file not found!")
        print(f"Expected: {data_dir / 'sila_input_amyloid.csv'}")
        return False

    # Extract variables
    age = data['AGE'].values
    value = data['AMY_GLOBAL_COMPOSITE'].values
    subid = data['RID'].values

    # SILA parameters (from sila_demo.m)
    dt = 0.25
    val0 = 0.79
    maxi = 200

    # Run Python implementation
    print("\nRunning Python SILA...")
    try:
        tsila_py, tdrs_py = sila(age, value, subid, dt, val0, maxi)
        print(f"  tsila shape: {tsila_py.shape}")
        print(f"  tdrs shape: {tdrs_py.shape}")

        test_py = sila_estimate(tsila_py, age, value, subid)
        print(f"  estimates shape: {test_py.shape}")
    except Exception as e:
        print(f"ERROR running Python SILA: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Load MATLAB results
    print("\nLoading MATLAB reference results...")
    try:
        tsila_ml = pd.read_csv(validation_dir / 'matlab_tsila.csv')
        tdrs_ml = pd.read_csv(validation_dir / 'matlab_tdrs.csv')
        test_ml = pd.read_csv(validation_dir / 'matlab_estimates.csv')
        print(f"  MATLAB tsila: {tsila_ml.shape}")
        print(f"  MATLAB tdrs: {tdrs_ml.shape}")
        print(f"  MATLAB estimates: {test_ml.shape}")
    except FileNotFoundError:
        print("WARNING: MATLAB reference files not found!")
        print("Please run the MATLAB validation script first.")
        print("\nSaving Python results for manual comparison...")

        tsila_py.to_csv(validation_dir / 'python_tsila.csv', index=False)
        tdrs_py.to_csv(validation_dir / 'python_tdrs.csv', index=False)
        test_py.to_csv(validation_dir / 'python_estimates.csv', index=False)

        print("\nPython results saved to validation/ directory")
        return None

    # Save Python results for debugging
    tsila_py.to_csv(validation_dir / 'python_tsila.csv', index=False)
    tdrs_py.to_csv(validation_dir / 'python_tdrs.csv', index=False)
    test_py.to_csv(validation_dir / 'python_estimates.csv', index=False)

    # Compare results
    all_results = []

    # Compare tsila
    result_tsila = validate_results(tsila_ml, tsila_py, 'tsila (integrated curve)')
    all_results.append(result_tsila)

    # Compare tdrs
    result_tdrs = validate_results(tdrs_ml, tdrs_py, 'tdrs (discrete rates)')
    all_results.append(result_tdrs)

    # Compare estimates
    result_estimates = validate_results(test_ml, test_py, 'estimates (subject-level)')
    all_results.append(result_estimates)

    # Summary
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    all_passed = all(r['pass'] for r in all_results)

    for r in all_results:
        status = "PASS" if r['pass'] else "FAIL"
        print(f"\n{r['table']:25s}: [{status}]")
        print(f"  Max absolute error: {r['max_abs_error']:.2e}")
        print(f"  Max relative error: {r['max_rel_error']:.2e}")
        print(f"  RMSE:              {r['rmse']:.2e}")
        print(f"  Correlation:        {r['correlation']:.8f}")

        if r['failed_columns']:
            print(f"  Failed columns: {', '.join(r['failed_columns'])}")

    print("\n" + "="*70)
    if all_passed:
        print("FINAL VERDICT: PASS")
        print("Python implementation matches MATLAB within tolerance (< 1e-10)")
    else:
        print("FINAL VERDICT: FAIL")
        print("Some columns exceed acceptable error tolerance")
    print("="*70 + "\n")

    return all_passed


if __name__ == "__main__":
    result = main()
    if result is True:
        sys.exit(0)
    elif result is False:
        sys.exit(1)
    else:
        sys.exit(2)  # Need MATLAB reference
