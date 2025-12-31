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
    if T <= 0:
        return float('nan')
    
    DD = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
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
    if T <= 0:
        return float('nan')
    
    d2 = (np.log(V / D) + (r - 0.5 * sigma_V**2) * T) / (sigma_V * np.sqrt(T))
    PD = norm.cdf(-d2)
    return max(0.0, min(1.0, PD)) 

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

