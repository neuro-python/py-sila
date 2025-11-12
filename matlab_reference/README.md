# MATLAB Reference Implementation

This directory contains the original MATLAB implementation of SILA for reference and validation purposes.

## Original Source

- **Authors**: Tobey J. Betthauser, PhD
- **Institution**: University of Wisconsin-Madison, Department of Medicine, Division of Geriatrics
- **Email**: tjbetthauser@medicine.wisc.edu
- **Paper**: Betthauser, T.J., et al. (2022). Multi-method investigation of factors influencing amyloid onset and impairment in three cohorts. *Brain*, 145(11), 4059-4071.
- **DOI**: 10.1093/brain/awac213

## Contents

- `SILA-AD-Biomarker-main/` - Complete MATLAB toolbox
  - `SILA.m` - Main SILA wrapper with kernel optimization
  - `ILLA.m` - Iterative Local Linear Approximation core
  - `SILA_estimate.m` - Subject-level estimation
  - `SILA_estimate_*.m` - Additional estimation utilities
  - `demo/` - Demonstration scripts
  - `toolbox/` - Dependencies (BCT toolbox)

## License

The MATLAB code is provided by the original authors for research purposes. Please refer to the original publication and contact the authors for licensing information.

## Usage

This MATLAB code is included for:

1. **Reference** during Python development
2. **Validation** and numerical comparison
3. **Understanding** the original algorithm implementation

**Note**: Users of py-sila do NOT need MATLAB installed. The Python implementation is standalone and fully functional.

## Running MATLAB Code (Optional)

If you have MATLAB and want to run the original code:

```matlab
% Add to MATLAB path
addpath('SILA-AD-Biomarker-main');

% Load your data
data = readtable('your_data.csv');

% Run SILA
[tsila, tdrs] = SILA(data.age, data.value, data.subid, 0.25, 0.79, 200);

% Get estimates
estimates = SILA_estimate(tsila, data.age, data.value, data.subid);
```

## Validation

The Python implementation was validated against this MATLAB code. See [`../validation/`](../validation/) for:
- Validation scripts
- Numerical comparison results (< 1e-10 error)
- Visualization of MATLAB vs Python outputs
