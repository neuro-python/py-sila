"""
SILA (Sampled Iterative Local Approximation) Toolbox
Python implementation with exact numerical equivalence to MATLAB

Reference:
Betthauser, T.J., et al. (2022). Multi-method investigation of factors
influencing amyloid onset and impairment in three cohorts. Brain, 145(11), 4059-4071.
"""

from .illa import illa
from .sila import sila
from .estimate import sila_estimate

__version__ = "1.0.0"
__all__ = ["illa", "sila", "sila_estimate"]
