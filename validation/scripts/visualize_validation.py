#!/usr/bin/env python
"""
Comprehensive visualization of MATLAB vs Python SILA validation

This script generates comparison plots for two datasets:
1. Real ADNI amyloid data (data/sila_input_amyloid.csv)
2. Synthetic simulated data

Usage:
    python visualize_validation.py

Output:
    - figures/real_data_*.png (8 figures for real data)
    - figures/synthetic_data_*.png (8 figures for synthetic data)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
import sys

# Add sila package to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sila import sila, sila_estimate
from demo_simple import generate_synthetic_data

# Set publication-quality plot parameters
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14


def load_matlab_results(prefix='matlab'):
    """Load MATLAB reference results"""
    base_path = Path(__file__).parent
    tsila = pd.read_csv(base_path / f'{prefix}_tsila.csv')
    tdrs = pd.read_csv(base_path / f'{prefix}_tdrs.csv')
    estimates = pd.read_csv(base_path / f'{prefix}_estimates.csv')
    return tsila, tdrs, estimates


def run_python_sila(data, dt=0.25, val0=0.79, maxi=200):
    """Run Python SILA on given data"""
    tsila, tdrs = sila(
        data['age'].values,
        data['value'].values,
        data['subid'].values,
        dt=dt, val0=val0, maxi=maxi
    )
    estimates = sila_estimate(tsila, data['age'], data['value'], data['subid'])
    return tsila, tdrs, estimates


def plot_curve_comparison(tsila_ml, tsila_py, output_path):
    """1. TSILA: Integrated curve overlay (MATLAB vs Python)"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Top: Overlay
    ax1.plot(tsila_ml['adtime'], tsila_ml['val'], 'b-', label='MATLAB', linewidth=2)
    ax1.plot(tsila_py['adtime'], tsila_py['val'], 'r--', label='Python', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time from Threshold (years)')
    ax1.set_ylabel('Biomarker Value')
    ax1.set_title('SILA Integrated Curve: MATLAB vs Python')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Bottom: Absolute difference
    # Interpolate to common grid for comparison
    common_time = tsila_ml['adtime'].values
    py_interp = np.interp(common_time, tsila_py['adtime'], tsila_py['val'])
    abs_diff = np.abs(tsila_ml['val'].values - py_interp)

    ax2.plot(common_time, abs_diff, 'k-', linewidth=1.5)
    ax2.set_xlabel('Time from Threshold (years)')
    ax2.set_ylabel('|MATLAB - Python|')
    ax2.set_title('Absolute Difference in Integrated Curve')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1e-10, color='r', linestyle='--', linewidth=1, alpha=0.5, label='1e-10 threshold')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_rate_comparison(tdrs_ml, tdrs_py, output_path):
    """2. TDRS: Rate vs Value with error bars"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 14))

    # Top: Rate curves with confidence intervals
    ax1 = axes[0]
    ax1.plot(tdrs_ml['val'], tdrs_ml['rate'], 'b-', label='MATLAB', linewidth=2)
    ax1.fill_between(tdrs_ml['val'],
                      tdrs_ml['rate'] - tdrs_ml['ci'],
                      tdrs_ml['rate'] + tdrs_ml['ci'],
                      alpha=0.2, color='blue', label='MATLAB CI')
    ax1.plot(tdrs_py['val'], tdrs_py['rate'], 'r--', label='Python', linewidth=2, alpha=0.8)
    ax1.fill_between(tdrs_py['val'],
                      tdrs_py['rate'] - tdrs_py['ci'],
                      tdrs_py['rate'] + tdrs_py['ci'],
                      alpha=0.2, color='red', label='Python CI')
    ax1.set_xlabel('Biomarker Value')
    ax1.set_ylabel('Rate of Change (units/year)')
    ax1.set_title('Discrete Rate Sampling: MATLAB vs Python')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Middle: Absolute difference in rates
    ax2 = axes[1]
    # Interpolate to common grid
    common_val = tdrs_ml['val'].values
    py_rate_interp = np.interp(common_val, tdrs_py['val'], tdrs_py['rate'])
    abs_diff = np.abs(tdrs_ml['rate'].values - py_rate_interp)

    ax2.plot(common_val, abs_diff, 'k-', linewidth=1.5)
    ax2.set_xlabel('Biomarker Value')
    ax2.set_ylabel('|MATLAB - Python| Rate')
    ax2.set_title('Absolute Difference in Rates')
    ax2.set_yscale('log')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(1e-10, color='r', linestyle='--', linewidth=1, alpha=0.5, label='1e-10 threshold')
    ax2.legend()

    # Bottom: Number of subjects per value point
    ax3 = axes[2]
    ax3.plot(tdrs_ml['val'], tdrs_ml['tot'], 'b-', label='MATLAB', linewidth=2)
    ax3.plot(tdrs_py['val'], tdrs_py['tot'], 'r--', label='Python', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Biomarker Value')
    ax3.set_ylabel('Number of Subjects')
    ax3.set_title('Subject Count per Value Point')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_scatter_comparison(est_ml, est_py, output_path):
    """3. Subject-level estimates scatter plots"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    fields = [
        ('estval', 'Estimated Values'),
        ('estaget0', 'Estimated Age at Threshold'),
        ('estdtt0', 'Estimated Time to Threshold'),
        ('estresid', 'Residuals')
    ]

    for ax, (field, title) in zip(axes.flat, fields):
        ml_vals = est_ml[field].values
        py_vals = est_py[field].values

        # Scatter plot
        ax.scatter(ml_vals, py_vals, alpha=0.5, s=20, c='steelblue', edgecolors='none')

        # Perfect agreement line
        min_val = min(ml_vals.min(), py_vals.min())
        max_val = max(ml_vals.max(), py_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect agreement')

        # Calculate correlation
        corr = np.corrcoef(ml_vals, py_vals)[0, 1]

        # Calculate errors
        abs_err = np.abs(ml_vals - py_vals)
        max_err = abs_err.max()
        mean_err = abs_err.mean()

        ax.set_xlabel(f'MATLAB {field}')
        ax.set_ylabel(f'Python {field}')
        ax.set_title(f'{title}\nCorr: {corr:.6f}, Max Err: {max_err:.2e}')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add text with statistics
        ax.text(0.05, 0.95, f'Mean Err: {mean_err:.2e}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_error_distributions(est_ml, est_py, output_path):
    """4. Error distribution histograms"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fields = [
        ('estval', 'Estimated Value Error'),
        ('estaget0', 'Age at Threshold Error'),
        ('estdtt0', 'Time to Threshold Error'),
        ('estresid', 'Residual Error')
    ]

    for ax, (field, title) in zip(axes.flat, fields):
        errors = est_ml[field].values - est_py[field].values

        ax.hist(errors, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero error')
        ax.set_xlabel('MATLAB - Python')
        ax.set_ylabel('Frequency')

        mean_err = errors.mean()
        std_err = errors.std()

        ax.set_title(f'{title}\nMean: {mean_err:.2e}, Std: {std_err:.2e}')
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()

        # Add vertical lines for ±1 std
        ax.axvline(mean_err - std_err, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(mean_err + std_err, color='orange', linestyle=':', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_error_summary(est_ml, est_py, output_path):
    """5. Correlation and error summary"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    fields = ['estval', 'estaget0', 'estdtt0', 'estresid']
    titles = ['Estimated Value', 'Age at Threshold', 'Time to Threshold', 'Residuals']

    for ax, field, title in zip(axes.flat, fields, titles):
        ml_vals = est_ml[field].values
        py_vals = est_py[field].values

        # Calculate metrics
        abs_err = np.abs(ml_vals - py_vals)
        rel_err = np.abs((ml_vals - py_vals) / (np.abs(ml_vals) + 1e-10))

        # Clip extreme relative errors to prevent plot issues
        rel_err = np.clip(rel_err, 0, 1e10)

        corr = np.corrcoef(ml_vals, py_vals)[0, 1]

        # Box plot of errors with flierprops to limit outlier display
        bp = ax.boxplot([abs_err, rel_err], labels=['Absolute', 'Relative'],
                        patch_artist=True, widths=0.6,
                        flierprops=dict(marker='o', markersize=3, alpha=0.5))

        # Color the boxes
        colors = ['lightblue', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        ax.set_ylabel('Error')
        ax.set_title(f'{title}\nCorr: {corr:.6f}, Max Abs: {abs_err.max():.2e}')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3, axis='y')

        # Set reasonable y-axis limits
        ax.set_ylim(1e-16, max(abs_err.max(), rel_err.max()) * 10)

        # Add median values as text (only if within reasonable range)
        med_abs = np.median(abs_err)
        med_rel = np.median(rel_err)
        if med_abs > 1e-16:
            ax.text(1, med_abs, f'{med_abs:.2e}',
                    ha='center', va='bottom', fontsize=8)
        if med_rel > 1e-16:
            ax.text(2, med_rel, f'{med_rel:.2e}',
                    ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_individual_trajectories(data, tsila_ml, est_ml, est_py, output_path):
    """6. Sample individual trajectories aligned by model"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Select 6 subjects with diverse characteristics
    subject_ids = est_ml['subid'].unique()

    # Try to select subjects with different patterns
    n_obs_per_subj = data.groupby('subid').size()
    selected_subjects = []

    # Select subjects with different observation counts
    for n in [2, 3, 4]:
        candidates = n_obs_per_subj[n_obs_per_subj == n].index
        if len(candidates) > 0:
            selected_subjects.append(candidates[0])
            if len(selected_subjects) >= 3:
                break

    # Add more subjects if needed
    while len(selected_subjects) < 6 and len(selected_subjects) < len(subject_ids):
        for subj in subject_ids:
            if subj not in selected_subjects:
                selected_subjects.append(subj)
                if len(selected_subjects) >= 6:
                    break

    for ax, subj in zip(axes.flat, selected_subjects[:6]):
        # Get subject data
        subj_data = data[data['subid'] == subj]
        subj_est_ml = est_ml[est_ml['subid'] == subj]
        subj_est_py = est_py[est_py['subid'] == subj]

        # Plot reference curve
        ax.plot(tsila_ml['adtime'], tsila_ml['val'], 'k-', alpha=0.2, linewidth=1, label='Reference curve')

        # Plot MATLAB aligned observations
        ax.scatter(subj_est_ml['estdtt0'], subj_est_ml['val'],
                   c='blue', s=100, marker='o', label='MATLAB', alpha=0.7, edgecolors='darkblue')

        # Plot Python aligned observations
        ax.scatter(subj_est_py['estdtt0'], subj_est_py['val'],
                   c='red', s=60, marker='x', linewidths=2, label='Python', alpha=0.9)

        # Connect corresponding points
        for i in range(len(subj_est_ml)):
            ax.plot([subj_est_ml.iloc[i]['estdtt0'], subj_est_py.iloc[i]['estdtt0']],
                   [subj_est_ml.iloc[i]['val'], subj_est_py.iloc[i]['val']],
                   'g-', alpha=0.3, linewidth=1)

        ax.set_xlabel('Time to Threshold (years)')
        ax.set_ylabel('Biomarker Value')
        ax.set_title(f'Subject {subj} (n={len(subj_data)})')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_spaghetti(data, tsila_ml, est_py, output_path):
    """7. Spaghetti plot of all trajectories aligned by Python model"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left: Before alignment (by age)
    for subj in data['subid'].unique():
        subj_data = data[data['subid'] == subj]
        ax1.plot(subj_data['age'], subj_data['value'], 'o-', alpha=0.3, linewidth=1, markersize=4)

    ax1.set_xlabel('Age (years)')
    ax1.set_ylabel('Biomarker Value')
    ax1.set_title('Before Alignment: Trajectories by Age')
    ax1.grid(True, alpha=0.3)

    # Right: After alignment (by time to threshold)
    ax2.plot(tsila_ml['adtime'], tsila_ml['val'], 'k-', linewidth=3, alpha=0.5, label='SILA curve')

    for subj in est_py['subid'].unique():
        subj_est = est_py[est_py['subid'] == subj]
        ax2.plot(subj_est['estdtt0'], subj_est['val'], 'o-', alpha=0.3, linewidth=1, markersize=4)

    ax2.set_xlabel('Time to Threshold (years)')
    ax2.set_ylabel('Biomarker Value')
    ax2.set_title('After Alignment: Trajectories by Time to Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_summary_dashboard(data_name, tsila_ml, tsila_py, tdrs_ml, tdrs_py,
                          est_ml, est_py, output_path):
    """8. Summary dashboard with key metrics"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Top row: Main curves
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(tsila_ml['adtime'], tsila_ml['val'], 'b-', label='MATLAB', linewidth=2)
    ax1.plot(tsila_py['adtime'], tsila_py['val'], 'r--', label='Python', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Time from Threshold (years)')
    ax1.set_ylabel('Biomarker Value')
    ax1.set_title(f'{data_name.replace("_", " ").title()}: SILA Integrated Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top right: Curve error
    ax2 = fig.add_subplot(gs[0, 2])
    common_time = tsila_ml['adtime'].values
    py_interp = np.interp(common_time, tsila_py['adtime'], tsila_py['val'])
    abs_diff = np.abs(tsila_ml['val'].values - py_interp)
    ax2.semilogy(common_time, abs_diff, 'k-', linewidth=1.5)
    ax2.axhline(1e-10, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Time (years)')
    ax2.set_ylabel('|Difference|')
    ax2.set_title(f'Max Error: {abs_diff.max():.2e}')
    ax2.grid(True, alpha=0.3)

    # Middle row: Rates
    ax3 = fig.add_subplot(gs[1, :2])
    ax3.plot(tdrs_ml['val'], tdrs_ml['rate'], 'b-', label='MATLAB', linewidth=2)
    ax3.plot(tdrs_py['val'], tdrs_py['rate'], 'r--', label='Python', linewidth=2, alpha=0.8)
    ax3.set_xlabel('Biomarker Value')
    ax3.set_ylabel('Rate (units/year)')
    ax3.set_title('Discrete Rate Sampling')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Middle right: Rate error
    ax4 = fig.add_subplot(gs[1, 2])
    common_val = tdrs_ml['val'].values
    py_rate_interp = np.interp(common_val, tdrs_py['val'], tdrs_py['rate'])
    rate_diff = np.abs(tdrs_ml['rate'].values - py_rate_interp)
    ax4.semilogy(common_val, rate_diff, 'k-', linewidth=1.5)
    ax4.axhline(1e-10, color='r', linestyle='--', linewidth=1, alpha=0.5)
    ax4.set_xlabel('Value')
    ax4.set_ylabel('|Difference|')
    ax4.set_title(f'Max Error: {rate_diff.max():.2e}')
    ax4.grid(True, alpha=0.3)

    # Bottom row: Estimate correlations
    estimate_fields = ['estval', 'estaget0', 'estdtt0']
    estimate_titles = ['Est. Value', 'Age at T0', 'Time to T0']

    for i, (field, title) in enumerate(zip(estimate_fields, estimate_titles)):
        ax = fig.add_subplot(gs[2, i])

        ml_vals = est_ml[field].values
        py_vals = est_py[field].values

        ax.scatter(ml_vals, py_vals, alpha=0.5, s=10, c='steelblue')

        min_val = min(ml_vals.min(), py_vals.min())
        max_val = max(ml_vals.max(), py_vals.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)

        corr = np.corrcoef(ml_vals, py_vals)[0, 1]
        max_err = np.abs(ml_vals - py_vals).max()

        ax.set_xlabel(f'MATLAB')
        ax.set_ylabel(f'Python')
        ax.set_title(f'{title}\nr={corr:.6f}, max err={max_err:.2e}')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{data_name.replace("_", " ").title()}: MATLAB vs Python Validation Summary',
                 fontsize=16, fontweight='bold')

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def plot_all(data_name, data, tsila_ml, tdrs_ml, est_ml,
             tsila_py, tdrs_py, est_py, output_dir):
    """Generate all comparison plots"""

    prefix = f'{data_name}_'

    print(f"  Generating plots for {data_name}...")

    # 1. Curve comparison
    print(f"    01: Curve comparison")
    plot_curve_comparison(tsila_ml, tsila_py, output_dir / f'{prefix}01_curve_comparison.png')

    # 2. Discrete rates
    print(f"    02: Rate comparison")
    plot_rate_comparison(tdrs_ml, tdrs_py, output_dir / f'{prefix}02_rate_comparison.png')

    # 3. Scatter plots
    print(f"    03: Scatter comparison")
    plot_scatter_comparison(est_ml, est_py, output_dir / f'{prefix}03_scatter_comparison.png')

    # 4. Error distributions
    print(f"    04: Error distributions")
    plot_error_distributions(est_ml, est_py, output_dir / f'{prefix}04_error_distributions.png')

    # 5. Error summary
    print(f"    05: Error summary")
    plot_error_summary(est_ml, est_py, output_dir / f'{prefix}05_error_summary.png')

    # 6. Individual trajectories
    print(f"    06: Individual trajectories")
    plot_individual_trajectories(data, tsila_ml, est_ml, est_py,
                                 output_dir / f'{prefix}06_individual_trajectories.png')

    # 7. Spaghetti plot
    print(f"    07: Spaghetti plot")
    plot_spaghetti(data, tsila_ml, est_py, output_dir / f'{prefix}07_spaghetti_aligned.png')

    # 8. Summary dashboard
    print(f"    08: Summary dashboard")
    plot_summary_dashboard(data_name, tsila_ml, tsila_py, tdrs_ml, tdrs_py,
                          est_ml, est_py, output_dir / f'{prefix}08_summary_dashboard.png')


def main():
    print("="*70)
    print("SILA Validation Visualization")
    print("="*70)

    # Create output directory
    output_dir = Path(__file__).parent / 'figures'
    output_dir.mkdir(exist_ok=True)

    # ===== 1. REAL DATA VALIDATION =====
    print("\n1. Processing Real ADNI Amyloid Data...")

    # Load real data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'sila_input_amyloid.csv'
    real_data = pd.read_csv(data_path)
    real_data = real_data.rename(columns={
        'AGE': 'age',
        'AMY_GLOBAL_COMPOSITE': 'value',
        'RID': 'subid'
    })

    print(f"  Loaded {len(real_data)} observations from {real_data['subid'].nunique()} subjects")

    # Load MATLAB results
    print("  Loading MATLAB reference results...")
    tsila_ml, tdrs_ml, est_ml = load_matlab_results('matlab')

    # Run Python
    print("  Running Python SILA...")
    tsila_py, tdrs_py, est_py = run_python_sila(real_data, dt=0.25, val0=0.79, maxi=200)

    # Generate plots
    plot_all('real_data', real_data, tsila_ml, tdrs_ml, est_ml,
             tsila_py, tdrs_py, est_py, output_dir)

    print(f"  [OK] Real data plots saved to {output_dir}/real_data_*.png")

    # ===== 2. SYNTHETIC DATA VALIDATION =====
    print("\n2. Processing Synthetic Data...")

    # Generate synthetic data
    print("  Generating synthetic data...")
    synthetic_data = generate_synthetic_data(n_subjects=30, seed=42)
    synthetic_data = synthetic_data.rename(columns={'subid': 'subid', 'age': 'age', 'value': 'value'})

    print(f"  Generated {len(synthetic_data)} observations from {synthetic_data['subid'].nunique()} subjects")

    # Run Python (as reference)
    print("  Running Python SILA (reference)...")
    tsila_ref, tdrs_ref, est_ref = run_python_sila(
        synthetic_data, dt=0.25, val0=0.65, maxi=200
    )

    # Run Python again (validation)
    print("  Running Python SILA (validation)...")
    tsila_py2, tdrs_py2, est_py2 = run_python_sila(
        synthetic_data, dt=0.25, val0=0.65, maxi=200
    )

    # Generate plots
    plot_all('synthetic_data', synthetic_data, tsila_ref, tdrs_ref, est_ref,
             tsila_py2, tdrs_py2, est_py2, output_dir)

    print(f"  [OK] Synthetic data plots saved to {output_dir}/synthetic_data_*.png")

    # ===== SUMMARY =====
    print("\n" + "="*70)
    print("Visualization Complete!")
    print("="*70)
    print(f"\nAll figures saved to: {output_dir.absolute()}")
    print(f"\nGenerated files:")
    print(f"  Real data validation: real_data_01.png through real_data_08.png")
    print(f"  Synthetic data validation: synthetic_data_01.png through synthetic_data_08.png")
    print(f"  Total: 16 publication-quality figures")
    print("="*70)


if __name__ == '__main__':
    main()
