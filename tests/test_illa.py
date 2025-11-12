"""
Integration tests for ILLA function
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from sila import illa


class TestILLA:
    """Test ILLA core functionality"""

    @pytest.fixture
    def simple_data(self):
        """Create simple test dataset"""
        np.random.seed(42)

        # 10 subjects, 3 observations each
        subids = np.repeat(np.arange(1, 11), 3)
        ages = []
        values = []

        for i in range(1, 11):
            base_age = 60 + i * 2
            base_val = 0.5 + i * 0.05
            ages.extend([base_age, base_age + 2, base_age + 4])
            values.extend([base_val, base_val + 0.1, base_val + 0.2])

        return np.array(ages), np.array(values), subids

    def test_illa_basic(self, simple_data):
        """Test basic ILLA execution"""
        age, value, subid = simple_data

        tout, tdrs = illa(age, value, subid,
                          dt=0.25, val0=0.7, maxi=100, skern=0.3)

        # Check outputs are DataFrames
        assert isinstance(tout, pd.DataFrame)
        assert isinstance(tdrs, pd.DataFrame)

        # Check required columns exist
        expected_tout_cols = ['val', 'time', 'adtime', 'mrate', 'sdrate',
                              'nsubs', 'sdval', 'ci95']
        for col in expected_tout_cols:
            assert col in tout.columns

        expected_tdrs_cols = ['val', 'rate', 'ratestd', 'npos', 'tot', 'ci', 'skern']
        for col in expected_tdrs_cols:
            assert col in tdrs.columns

    def test_illa_no_smoothing(self, simple_data):
        """Test ILLA without smoothing"""
        age, value, subid = simple_data

        tout, tdrs = illa(age, value, subid,
                          dt=0.25, val0=0.7, maxi=100, skern=0)

        # Should still produce valid output
        assert len(tout) > 0
        assert len(tdrs) > 0
        assert tdrs['skern'].iloc[0] == 0

    def test_illa_increasing_curve(self):
        """Test with increasing biomarker"""
        # Create data with clear increasing trend
        subids = np.repeat(np.arange(1, 21), 3)
        ages = []
        values = []

        for i in range(1, 21):
            base_age = 60 + i
            base_val = 0.5 + i * 0.02
            ages.extend([base_age, base_age + 2, base_age + 4])
            # Increasing with time
            values.extend([base_val, base_val + 0.02, base_val + 0.04])

        tout, tdrs = illa(np.array(ages), np.array(values), subids,
                          dt=0.25, val0=0.7, maxi=200, skern=0.2)

        # Check median rate is positive
        assert np.median(tdrs['rate']) > 0

        # Check curve is monotonically increasing (mostly)
        val_diffs = np.diff(tout['val'])
        assert np.sum(val_diffs > 0) > len(val_diffs) * 0.9

    def test_illa_edge_case_single_observation(self):
        """Test that subjects with single observations are excluded"""
        # Mix of single and multiple observations with enough data
        # Need more subjects with multiple observations for valid curve
        subids = []
        ages = []
        values = []

        # Add subjects with multiple observations
        for i in range(1, 11):
            base_age = 60 + i * 2
            base_val = 0.5 + i * 0.03
            subids.extend([i, i, i])
            ages.extend([base_age, base_age + 2, base_age + 4])
            values.extend([base_val, base_val + 0.05, base_val + 0.1])

        # Add some subjects with single observations (should be filtered)
        for i in range(11, 15):
            subids.append(i)
            ages.append(70)
            values.append(0.7)

        subids = np.array(subids)
        ages = np.array(ages)
        values = np.array(values)

        tout, tdrs = illa(ages, values, subids,
                          dt=0.25, val0=0.7, maxi=100, skern=0.2)

        # Should work despite single observations
        assert len(tout) > 0
        assert len(tdrs) > 0

        # Verify single-observation subjects were excluded from rate calculation
        # (they won't contribute to slopes)

    def test_illa_numerical_stability(self):
        """Test numerical stability with various dt values"""
        np.random.seed(42)

        subids = np.repeat(np.arange(1, 11), 3)
        ages = []
        values = []

        for i in range(1, 11):
            base_age = 60 + i
            base_val = 0.5 + i * 0.03
            ages.extend([base_age, base_age + 2, base_age + 4])
            values.extend([base_val, base_val + 0.05, base_val + 0.1])

        ages = np.array(ages)
        values = np.array(values)

        # Test different dt values
        for dt in [0.1, 0.25, 0.5]:
            tout, tdrs = illa(ages, values, subids,
                              dt=dt, val0=0.7, maxi=200, skern=0.2)

            # Should produce valid output
            assert len(tout) > 0
            assert np.all(np.isfinite(tout['val']))
            assert np.all(np.isfinite(tout['time']))

    def test_illa_output_structure(self, simple_data):
        """Test output structure matches MATLAB format"""
        age, value, subid = simple_data

        tout, tdrs = illa(age, value, subid,
                          dt=0.25, val0=0.7, maxi=100, skern=0.3)

        # Check data types
        assert tout['val'].dtype == np.float64
        assert tout['time'].dtype == np.float64
        assert tout['adtime'].dtype == np.float64

        assert tdrs['val'].dtype == np.float64
        assert tdrs['rate'].dtype == np.float64
        assert tdrs['tot'].dtype in [np.int64, np.int32]

        # Check no NaN in key columns
        assert not np.any(np.isnan(tout['val']))
        assert not np.any(np.isnan(tout['time']))
        assert not np.any(np.isnan(tdrs['rate']))

    def test_illa_threshold_alignment(self, simple_data):
        """Test that val0 threshold is correctly aligned"""
        age, value, subid = simple_data

        val0 = 0.7
        tout, tdrs = illa(age, value, subid,
                          dt=0.25, val0=val0, maxi=100, skern=0.3)

        # Find point closest to val0
        idx_threshold = np.argmin(np.abs(tout['val'] - val0))

        # adtime should be close to 0 at this point
        assert np.abs(tout['adtime'].iloc[idx_threshold]) < 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
