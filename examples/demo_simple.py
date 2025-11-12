"""
Simple demo script to test SILA Python implementation
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add sila package to path
sys.path.insert(0, str(Path(__file__).parent))

from sila import sila, sila_estimate


def generate_synthetic_data(n_subjects=30, seed=42):
    """Generate synthetic longitudinal biomarker data"""
    np.random.seed(seed)

    data = []

    for subid in range(1, n_subjects + 1):
        # Random number of observations (2-4)
        n_obs = np.random.randint(2, 5)

        # Base characteristics
        base_age = np.random.uniform(60, 80)
        base_val = np.random.uniform(0.5, 0.9)
        rate = np.random.uniform(0.01, 0.04)  # Increasing with time

        for obs in range(n_obs):
            age = base_age + obs * np.random.uniform(1.5, 3)
            val = base_val + obs * rate + np.random.normal(0, 0.02)

            data.append({
                'subid': subid,
                'age': age,
                'value': val
            })

    df = pd.DataFrame(data)
    return df


def main():
    """Run simple demo"""
    print("="*70)
    print("SILA Python Implementation - Simple Demo")
    print("="*70)

    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(n_subjects=30)
    print(f"   Generated {len(df)} observations from {df['subid'].nunique()} subjects")
    print(f"   Age range: {df['age'].min():.1f} - {df['age'].max():.1f}")
    print(f"   Value range: {df['value'].min():.3f} - {df['value'].max():.3f}")

    # Extract variables
    age = df['age'].values
    value = df['value'].values
    subid = df['subid'].values

    # Run SILA
    print("\n2. Running SILA with automatic kernel optimization...")
    try:
        tsila, tdrs = sila(
            age, value, subid,
            dt=0.25,
            val0=0.7,
            maxi=200
        )

        print(f"   Integrated curve: {len(tsila)} points")
        print(f"   Discrete rates: {len(tdrs)} samples")
        print(f"   Optimal kernel: {tdrs['skern'].iloc[0]:.3f}")
        print(f"   Time range: {tsila['adtime'].min():.2f} to {tsila['adtime'].max():.2f} years")
        print(f"   Value range: {tsila['val'].min():.3f} to {tsila['val'].max():.3f}")

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Run SILA_estimate
    print("\n3. Generating subject-level estimates...")
    try:
        estimates = sila_estimate(
            tsila, age, value, subid,
            align_event='last',
            extrap_years=3,
            truncate_aget0='yes'
        )

        print(f"   Estimates: {len(estimates)} observations")

        # Calculate statistics
        rmse = np.sqrt(np.mean(estimates['estresid'] ** 2))
        mae = np.mean(np.abs(estimates['estresid']))
        n_positive = np.sum(estimates['estpos'])

        print(f"   RMSE: {rmse:.4f}")
        print(f"   MAE: {mae:.4f}")
        print(f"   Positive cases: {n_positive}/{len(estimates)} ({100*n_positive/len(estimates):.1f}%)")

        # Show sample estimates
        print("\n4. Sample estimates (first 5 observations):")
        print(estimates[['subid', 'age', 'val', 'estval', 'estaget0', 'estdtt0', 'estresid']].head())

    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Display curve characteristics
    print("\n5. Curve characteristics:")
    print(f"   Median rate: {np.median(tdrs['rate']):.4f} units/year")
    print(f"   Rate range: {tdrs['rate'].min():.4f} to {tdrs['rate'].max():.4f}")
    print(f"   Mean subjects per rate point: {tdrs['tot'].mean():.1f}")

    print("\n" + "="*70)
    print("Demo completed successfully!")
    print("="*70)

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
