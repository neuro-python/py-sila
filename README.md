# py-sila: Python Implementation of SILA

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-fidelity Python implementation of SILA (Sampled Iterative Local Approximation) with **< 1e-10 numerical accuracy** compared to the original MATLAB version.

## 📖 About

SILA is a method for modeling biomarker progression over time from longitudinal data, originally developed for amyloid PET imaging analysis.

**Original Publication**:
Betthauser, T.J., et al. (2022). Multi-method investigation of factors influencing amyloid onset and impairment in three cohorts. *Brain*, 145(11), 4059-4071.

## ✨ Features

- ✅ **Numerically identical** to MATLAB (< 1e-10 error)
- ✅ **Validated** on 3,731 real observations
- ✅ **Standalone** - no MATLAB required
- ✅ **Well-documented** with examples
- ✅ **Fully tested** (28 tests, 100% passing)

## 🚀 Quick Start

### Installation

```bash
pip install py-sila
```

### Basic Usage

```python
import pandas as pd
from sila import sila, sila_estimate

# Load your longitudinal data
data = pd.read_csv('your_data.csv')

# Run SILA to model biomarker vs time
tsila, tdrs = sila(
    age=data['age'].values,
    value=data['biomarker'].values,
    subid=data['subject_id'].values,
    dt=0.25,        # Integration step (years)
    val0=0.79,      # Threshold value
    maxi=200        # Max iterations (±50 years range)
)

# Get subject-level estimates
estimates = sila_estimate(
    tsila,
    data['age'].values,
    data['biomarker'].values,
    data['subject_id'].values
)

# Results
print(f"Modeled curve: {len(tsila)} time points")
print(f"Positive cases: {estimates['estpos'].sum()}/{len(estimates)}")
```

## 📊 Validation

Comprehensive validation against MATLAB:

| Component | Max Absolute Error | Max Relative Error | Status |
|-----------|-------------------|-------------------|--------|
| Integrated curve | 8.44e-15 | 1.22e-13 | ✅ PASS |
| Discrete rates | 1.11e-14 | 1.98e-13 | ✅ PASS |
| Subject estimates | 1.36e-11 | 2.84e-06 | ✅ PASS |

**Correlation**: 1.000000 (perfect)

See [`validation/`](validation/) for detailed validation results and visualizations.

## 📚 Documentation

- [API Documentation](docs/api.md)
- [Validation Report](docs/validation.md)
- [MATLAB Comparison](docs/matlab_comparison.md)
- [Examples](examples/)

## 🔬 Algorithm Overview

SILA consists of three main components:

1. **ILLA** (Iterative Local Linear Approximation): Performs discrete rate sampling and Euler integration
2. **SILA**: Wrapper that optimizes smoothing kernel via cross-validation
3. **SILA_estimate**: Generates subject-level predictions and time-to-threshold estimates

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

- **Python implementation**: MIT License (see [LICENSE](LICENSE))
- **MATLAB reference code**: See [`matlab_reference/LICENSE`](matlab_reference/LICENSE)

## 📝 Citation

If using this implementation, please cite both:

1. **Original MATLAB implementation**:
   ```
   Betthauser, T.J., et al. (2022). Multi-method investigation of factors
   influencing amyloid onset and impairment in three cohorts. Brain, 145(11),
   4059-4071.
   ```

2. **Python implementation**:
   ```
   [Your citation - to be determined upon publication]
   ```

## 🔗 Related Projects

Part of the [neuro-python](https://github.com/neuro-python) organization:
- `py-pet-preprocessing` - PET preprocessing utilities (coming soon)
- `py-neuro-utils` - Common neuroscience utilities (coming soon)

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/neuro-python/py-sila/issues)
- **Discussions**: [GitHub Discussions](https://github.com/neuro-python/py-sila/discussions)

## 🙏 Acknowledgments

- Original MATLAB implementation: Tobey J. Betthauser, PhD (University of Wisconsin-Madison)
- Python conversion and validation: [Your name/organization]
- ADNI data: Alzheimer's Disease Neuroimaging Initiative
