"""
Risk Measures: Distance-to-Default and Default Probability

Compute credit risk measures from calibrated asset parameters.
"""

import numpy as np
from scipy.stats import norm


def distance_to_default(V, D, T, r, sigma_V):
    """
    Calculate distance-to-default (DD).
    
    DD measures how many standard deviations the asset value is above
    the default threshold.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    float
        Distance-to-default
    """
    # TODO: Implement distance-to-default calculation
    # Hint: See Mathematical Background section in README.md
    # DD = (E[V_T] - D) / std(V_T)
    # where E[V_T] = V * exp(r*T)
    # and std(V_T) = V * exp(r*T) * sqrt(exp(sigma_V^2*T) - 1)
    
    if T <= 0:
        return float('inf') if V > D else float('-inf')
    
    E_VT = V * np.exp(r * T)
    std_VT = V * np.exp(r * T) * np.sqrt(np.exp(sigma_V**2 * T) - 1)
    
    if std_VT <= 0:
        return float('inf') if E_VT > D else float('-inf')
    
    DD = (E_VT - D) / std_VT
    return DD


def default_probability(V, D, T, r, sigma_V):
    """
    Calculate risk-neutral default probability (PD).
    
    The probability that asset value at maturity will be below debt.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    float
        Default probability (between 0 and 1)
    """
    # TODO: Implement default probability calculation
    # Hint: See Mathematical Background section in README.md
    # PD = Phi(-d2) where d2 = (ln(V/D) + (r - sigma_V^2/2)*T) / (sigma_V*sqrt(T))
    
    if T <= 0 or sigma_V <= 0 or V <= 0 or D <= 0:
        return 1.0 if V < D else 0.0
    
    d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    PD = norm.cdf(-d2)
    
    return max(0.0, min(1.0, PD))  # Ensure PD is between 0 and 1


def compute_risk_measures(V, D, T, r, sigma_V):
    """
    Compute both distance-to-default and default probability.
    
    Parameters:
    -----------
    V : float
        Current asset value
    D : float
        Face value of debt
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma_V : float
        Asset volatility (annualized)
    
    Returns:
    --------
    dict
        Dictionary with 'DD' and 'PD' keys
    """
    DD = distance_to_default(V, D, T, r, sigma_V)
    PD = default_probability(V, D, T, r, sigma_V)
    
    return {
        'DD': DD,
        'PD': PD
    }

