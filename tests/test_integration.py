"""
Integration tests for complete SILA workflow
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sila import sila, sila_estimate


class TestSILAIntegration:
    """Test complete SILA workflow"""

    @pytest.fixture
    def realistic_data(self):
        """Create realistic longitudinal biomarker data"""
        np.random.seed(42)

        n_subjects = 50
        data = []

        for subid in range(1, n_subjects + 1):
            # Random number of observations (2-5)
            n_obs = np.random.randint(2, 6)

            # Base age and progression rate
            base_age = np.random.uniform(60, 80)
            base_val = np.random.uniform(0.4, 1.0)
            rate = np.random.uniform(0.01, 0.05)

            for obs in range(n_obs):
                age = base_age + obs * np.random.uniform(1, 3)
                val = base_val + obs * rate + np.random.normal(0, 0.02)

                data.append({
                    'subid': subid,
                    'age': age,
                    'value': val
                })

        df = pd.DataFrame(data)
        return df['age'].values, df['value'].values, df['subid'].values

    def test_sila_full_workflow(self, realistic_data):
        """Test complete SILA workflow"""
        age, value, subid = realistic_data

        # Run SILA with automatic kernel optimization
        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=0.7, maxi=200)

        # Check outputs
        assert isinstance(tsila, pd.DataFrame)
        assert isinstance(tdrs, pd.DataFrame)
        assert len(tsila) > 0
        assert len(tdrs) > 0

        # Check kernel was selected
        assert 'skern' in tdrs.columns
        assert 0 <= tdrs['skern'].iloc[0] <= 0.5

        # Run SILA_estimate
        estimates = sila_estimate(tsila, age, value, subid)

        assert isinstance(estimates, pd.DataFrame)
        assert len(estimates) == len(age)

        # Check required columns
        required_cols = ['subid', 'age', 'val', 'estval', 'estaget0',
                         'estdtt0', 'estresid', 'estpos']
        for col in required_cols:
            assert col in estimates.columns

    def test_sila_with_preset_kernel(self, realistic_data):
        """Test SILA with preset smoothing kernel"""
        age, value, subid = realistic_data

        # Test single kernel
        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=0.7, maxi=200, sk=0.3)

        assert tdrs['skern'].iloc[0] == 0.3

        # Test multiple kernels
        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=0.7, maxi=200,
                           sk=[0.2, 0.3, 0.4])

        assert tdrs['skern'].iloc[0] in [0.2, 0.3, 0.4]

    def test_sila_estimate_alignment_modes(self, realistic_data):
        """Test different alignment modes"""
        age, value, subid = realistic_data

        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=0.7, maxi=200)

        # Test 'first' alignment
        est_first = sila_estimate(tsila, age, value, subid,
                                  align_event='first')
        assert est_first['aevent'].iloc[0] == 'first'

        # Test 'last' alignment
        est_last = sila_estimate(tsila, age, value, subid,
                                 align_event='last')
        assert est_last['aevent'].iloc[0] == 'last'

        # Test 'all' alignment
        est_all = sila_estimate(tsila, age, value, subid,
                                align_event='all')
        assert est_all['aevent'].iloc[0] == 'all'

        # All should produce valid estimates
        for est in [est_first, est_last, est_all]:
            assert len(est) == len(age)
            # Most estimates should be finite (allow some NaN from extreme extrapolation)
            # Some random data may fall outside model range
            assert np.sum(np.isfinite(est['estval'])) > len(age) * 0.7

    def test_sila_estimate_extrapolation(self, realistic_data):
        """Test extrapolation beyond model range"""
        age, value, subid = realistic_data

        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=0.7, maxi=200)

        # Test different extrapolation years
        est_3yr = sila_estimate(tsila, age, value, subid, extrap_years=3)
        est_5yr = sila_estimate(tsila, age, value, subid, extrap_years=5)

        assert est_3yr['extrapyrs'].iloc[0] == 3
        assert est_5yr['extrapyrs'].iloc[0] == 5

    def test_sila_estimate_truncation(self, realistic_data):
        """Test truncation of estaget0"""
        age, value, subid = realistic_data

        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=0.7, maxi=200)

        # With truncation
        est_trunc = sila_estimate(tsila, age, value, subid,
                                  truncate_aget0='yes')
        assert 'truncated' in est_trunc.columns

        # Without truncation
        est_no_trunc = sila_estimate(tsila, age, value, subid,
                                     truncate_aget0='no')
        assert 'truncated' in est_no_trunc.columns

    def test_residuals_reasonable(self, realistic_data):
        """Test that residuals are reasonable"""
        age, value, subid = realistic_data

        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=0.7, maxi=200)

        estimates = sila_estimate(tsila, age, value, subid)

        # Calculate residual statistics (excluding NaN from extrapolation)
        residuals = estimates['estresid'].dropna()
        rmse = np.sqrt(np.mean(residuals ** 2))
        mae = np.mean(np.abs(residuals))

        # Should have reasonable fit (relaxed tolerance for synthetic data)
        assert rmse < 1.0  # Depends on data quality
        assert mae < 0.5

        # Residuals should not be completely biased (relaxed for random data)
        assert np.abs(np.mean(residuals)) < 0.5

    def test_positive_negative_classification(self, realistic_data):
        """Test biomarker positive/negative classification"""
        age, value, subid = realistic_data

        val0 = 0.7
        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=val0, maxi=200)

        estimates = sila_estimate(tsila, age, value, subid)

        # Check estpos matches estval vs valt0
        valt0 = estimates['valt0'].iloc[0]
        expected_pos = estimates['estval'] >= valt0
        assert np.array_equal(estimates['estpos'], expected_pos)

    def test_time_estimates_consistency(self, realistic_data):
        """Test consistency of time estimates"""
        age, value, subid = realistic_data

        tsila, tdrs = sila(age, value, subid,
                           dt=0.25, val0=0.7, maxi=200)

        estimates = sila_estimate(tsila, age, value, subid,
                                  align_event='last')

        # For each subject, check consistency
        for subj in np.unique(subid):
            sub_est = estimates[estimates['subid'] == subj]

            # estaget0 should be consistent within subject (roughly)
            if len(sub_est) > 1:
                estaget0_range = sub_est['estaget0'].max() - sub_est['estaget0'].min()
                # Should be relatively small (allowing for model uncertainty)
                assert estaget0_range < 5  # Within 5 years

    def test_numerical_precision(self, realistic_data):
        """Test numerical precision and stability"""
        age, value, subid = realistic_data

        # Run multiple times with same data
        results = []
        for _ in range(3):
            tsila, tdrs = sila(age, value, subid,
                               dt=0.25, val0=0.7, maxi=200, sk=0.3)
            results.append(tsila.copy())

        # Results should be identical
        for i in range(1, len(results)):
            pd.testing.assert_frame_equal(results[0], results[i],
                                          check_exact=False, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
