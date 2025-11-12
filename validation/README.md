# Validation Framework

This directory contains tools and data for validating the Python implementation against the original MATLAB code.

## Quick Validation

### Option 1: Use Pre-computed MATLAB Results (Recommended - No MATLAB required)

```bash
cd scripts
python compare_results.py
python visualize_validation.py
```

This compares Python outputs against pre-computed MATLAB reference results stored in `matlab_reference_results/`.

### Option 2: Full Re-validation (Requires MATLAB)

```bash
# 1. Generate fresh MATLAB results
cd scripts
matlab -batch "generate_matlab_reference"

# 2. Compare with Python
python compare_results.py

# 3. Generate visualizations
python visualize_validation.py
```

## Validation Results

Current validation status on **3,731 real observations** from ADNI:

| Component | Max Abs Error | Max Rel Error | RMSE | Correlation | Status |
|-----------|---------------|---------------|------|-------------|--------|
| **Integrated curve** (tsila) | 8.44e-15 | 1.22e-13 | 3.72e-15 | 1.000000 | ✅ PASS |
| **Discrete rates** (tdrs) | 1.11e-14 | 1.98e-13 | 4.57e-15 | 1.000000 | ✅ PASS |
| **Subject estimates** | 1.36e-11 | 2.84e-06 | 7.80e-12 | 1.000000 | ✅ PASS |

**Verdict**: Python implementation matches MATLAB within tolerance (< 1e-10) ✅

## Directory Structure

```
validation/
├── README.md                      # This file
├── data/                          # Validation datasets
│   ├── sila_input_amyloid.csv     # Real ADNI amyloid PET data (3,731 obs)
│   └── filtered_data.csv          # Filtered dataset
├── matlab_reference_results/      # Pre-computed MATLAB outputs
│   ├── matlab_tsila.csv           # MATLAB integrated curve
│   ├── matlab_tdrs.csv            # MATLAB discrete rates
│   └── matlab_estimates.csv       # MATLAB subject estimates
├── scripts/                       # Validation scripts
│   ├── generate_matlab_reference.m  # MATLAB script to generate reference
│   ├── compare_results.py           # Python vs MATLAB comparison
│   └── visualize_validation.py      # Visualization generation
└── figures/                       # Validation visualizations (16 plots)
    ├── real_data_01_curve_comparison.png
    ├── real_data_02_rate_comparison.png
    ├── real_data_03_scatter_comparison.png
    ├── real_data_04_error_distributions.png
    ├── real_data_05_error_summary.png
    ├── real_data_06_individual_trajectories.png
    ├── real_data_07_spaghetti_aligned.png
    ├── real_data_08_summary_dashboard.png
    └── synthetic_data_01-08.png
```

## Validation Methodology

1. **Real Data Validation**
   - Dataset: 3,731 observations from 1,712 subjects (ADNI amyloid PET)
   - Comparison: All outputs compared element-by-element
   - Metrics: Absolute error, relative error, RMSE, correlation

2. **Synthetic Data Validation**
   - Dataset: 96 observations from 30 simulated subjects
   - Purpose: Test reproducibility and edge cases
   - Expected: Perfect match (machine precision)

3. **Visual Validation**
   - 16 comparison plots (8 real + 8 synthetic)
   - Curve overlays, scatter plots, error distributions
   - Individual trajectory alignment

## Data

### sila_input_amyloid.csv

Real ADNI amyloid PET data with columns:
- `PTID`: Patient ID
- `RID`: Registry ID (subject identifier)
- `EXAMDATE`: Examination date
- `AGE`: Age at scan (years)
- `AMY_GLOBAL_COMPOSITE`: Amyloid centiloid value
- `AMY_TRACER`: PET tracer used (FBP/FBB)

**Source**: Alzheimer's Disease Neuroimaging Initiative (ADNI)
**Citation**: See [ADNI acknowledgment](http://adni.loni.usc.edu/)

## Validation Scripts

### generate_matlab_reference.m

Generates MATLAB reference outputs:
```matlab
% Load data
data = readtable('../../data/sila_input_amyloid.csv');

% Run MATLAB SILA
[tsila, tdrs] = SILA(data.AGE, data.AMY_GLOBAL_COMPOSITE, data.RID, 0.25, 0.79, 200);
test = SILA_estimate(tsila, data.AGE, data.AMY_GLOBAL_COMPOSITE, data.RID);

% Save results
writetable(tsila, '../matlab_reference_results/matlab_tsila.csv');
writetable(tdrs, '../matlab_reference_results/matlab_tdrs.csv');
writetable(test, '../matlab_reference_results/matlab_estimates.csv');
```

### compare_results.py

Compares Python vs MATLAB outputs:
- Loads MATLAB reference results
- Runs Python SILA on same data
- Calculates error metrics for all columns
- Reports PASS/FAIL for each component

### visualize_validation.py

Generates 16 comparison plots:
1. Curve comparison (MATLAB vs Python overlay)
2. Rate comparison with confidence intervals
3. Scatter plots (4 key variables)
4. Error distribution histograms
5. Error summary box plots
6. Individual subject trajectories
7. Spaghetti plot (all subjects aligned)
8. Summary dashboard

## Interpreting Results

### Error Types

**Absolute Error**: `|MATLAB - Python|`
- Should be < 1e-10 for perfect match
- Values < 1e-14 indicate machine precision

**Relative Error**: `|MATLAB - Python| / |MATLAB|`
- Can be larger when MATLAB value is near zero
- Focus on absolute error when relative is large

**RMSE**: Root Mean Square Error
- Overall error across all values
- Should be < 1e-10

**Correlation**: Pearson correlation coefficient
- Should be 1.000000 (perfect)
- Indicates perfect linear relationship

### Pass Criteria

✅ **PASS**: All of the following:
- Max absolute error < 1e-10
- Correlation > 0.999999
- Visual inspection shows perfect overlap

❌ **FAIL**: Any of:
- Max absolute error > 1e-10
- Systematic bias detected
- Visual discrepancies

## Troubleshooting

**Q: Python results differ from MATLAB**
A: Ensure you're using the same:
- Data file (check path)
- Parameters (dt=0.25, val0=0.79, maxi=200)
- Python version (3.8+)
- Package versions (see requirements.txt)

**Q: MATLAB script fails**
A: Check:
- MATLAB path includes `../../matlab_reference/SILA-AD-Biomarker-main`
- Data file exists at correct path
- MATLAB version (R2019b or later recommended)

**Q: Visualization shows large errors**
A: Check axes scales:
- Errors may look large due to zoom
- Actual errors are < 1e-10 (tiny!)
- See histogram centers, not spread

## Contact

For validation questions or issues:
- GitHub Issues: https://github.com/neuro-python/py-sila/issues
- Original MATLAB author: tjbetthauser@medicine.wisc.edu
