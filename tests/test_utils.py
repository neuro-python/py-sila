"""
Unit tests for utility functions
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sila.utils import rloess_smooth, polyfit_matlab, interp1_matlab, fitlm_matlab


class TestPolyfit:
    """Test polyfit MATLAB equivalence"""

    def test_linear_fit(self):
        """Test linear regression"""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float64)

        p = polyfit_matlab(x, y, 1)

        # Should get slope=2, intercept=0
        assert np.abs(p[0] - 2.0) < 1e-10
        assert np.abs(p[1] - 0.0) < 1e-10

    def test_quadratic_fit(self):
        """Test quadratic fit"""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = x ** 2

        p = polyfit_matlab(x, y, 2)

        # Should get [1, 0, 0] for y = x^2
        assert np.abs(p[0] - 1.0) < 1e-10
        assert np.abs(p[1] - 0.0) < 1e-10
        assert np.abs(p[2] - 0.0) < 1e-10

    def test_with_noise(self):
        """Test with noisy data"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3 + np.random.normal(0, 0.5, 50)

        p = polyfit_matlab(x, y, 1)

        # Should be close to slope=2, intercept=3
        assert np.abs(p[0] - 2.0) < 0.5
        assert np.abs(p[1] - 3.0) < 1.0


class TestInterp1:
    """Test 1D interpolation"""

    def test_linear_interpolation(self):
        """Test linear interpolation"""
        x = np.array([0, 1, 2, 3], dtype=np.float64)
        y = np.array([0, 2, 4, 6], dtype=np.float64)
        xi = np.array([0.5, 1.5, 2.5], dtype=np.float64)

        yi = interp1_matlab(x, y, xi, method='linear')

        expected = np.array([1.0, 3.0, 5.0])
        np.testing.assert_allclose(yi, expected, rtol=1e-10)

    def test_extrapolation_returns_nan(self):
        """Test that extrapolation returns NaN"""
        x = np.array([1, 2, 3], dtype=np.float64)
        y = np.array([10, 20, 30], dtype=np.float64)
        xi = np.array([0, 4], dtype=np.float64)

        yi = interp1_matlab(x, y, xi)

        assert np.isnan(yi[0])
        assert np.isnan(yi[1])

    def test_exact_match(self):
        """Test interpolation at exact points"""
        x = np.array([1, 2, 3, 4], dtype=np.float64)
        y = np.array([5, 10, 15, 20], dtype=np.float64)

        yi = interp1_matlab(x, y, x)

        np.testing.assert_allclose(yi, y, rtol=1e-10)


class TestFitlm:
    """Test linear model fitting"""

    def test_simple_linear(self):
        """Test simple linear fit"""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = np.array([2, 4, 6, 8, 10], dtype=np.float64)

        result = fitlm_matlab(x, y)

        # Should get intercept=0, slope=2
        assert np.abs(result['Estimate'][0] - 0.0) < 1e-10
        assert np.abs(result['Estimate'][1] - 2.0) < 1e-10

    def test_with_intercept(self):
        """Test linear fit with non-zero intercept"""
        x = np.array([1, 2, 3, 4, 5], dtype=np.float64)
        y = 3 * x + 5

        result = fitlm_matlab(x, y)

        assert np.abs(result['Estimate'][0] - 5.0) < 1e-10
        assert np.abs(result['Estimate'][1] - 3.0) < 1e-10


class TestLoess:
    """Test LOESS smoothing"""

    def test_smooth_linear_data(self):
        """Test that LOESS preserves linear trend"""
        x = np.linspace(0, 10, 50)
        y = 2 * x + 3

        ys = rloess_smooth(x, y, span=0.3)

        # Should closely match input for linear data
        np.testing.assert_allclose(ys, y, rtol=0.01)

    def test_smooth_noisy_data(self):
        """Test that LOESS smooths noisy data"""
        np.random.seed(42)
        x = np.linspace(0, 10, 100)
        y_true = 2 * x + 3
        y_noisy = y_true + np.random.normal(0, 1, 100)

        ys = rloess_smooth(x, y_noisy, span=0.3)

        # Smoothed should be closer to true than noisy
        error_noisy = np.mean((y_noisy - y_true) ** 2)
        error_smooth = np.mean((ys - y_true) ** 2)

        assert error_smooth < error_noisy

    def test_different_spans(self):
        """Test different smoothing spans"""
        np.random.seed(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + np.random.normal(0, 0.1, 50)

        ys_small = rloess_smooth(x, y, span=0.1)
        ys_large = rloess_smooth(x, y, span=0.5)

        # Larger span should produce smoother curve
        diff_small = np.sum(np.abs(np.diff(ys_small)))
        diff_large = np.sum(np.abs(np.diff(ys_large)))

        assert diff_large < diff_small

    def test_span_bounds(self):
        """Test invalid span values raise errors"""
        x = np.linspace(0, 10, 50)
        y = x ** 2

        with pytest.raises(ValueError):
            rloess_smooth(x, y, span=0)

        with pytest.raises(ValueError):
            rloess_smooth(x, y, span=1.5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
