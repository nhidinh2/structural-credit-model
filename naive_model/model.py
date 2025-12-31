"""
Merton Structural Credit Model

Implement the baseline Merton (1974) model here.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes formula for European call option price.
    
    Parameters:
    -----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns:
    --------
    float
        Call option price
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(S - K * np.exp(-r * T), 0)
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price



def black_scholes_delta(S, K, T, r, sigma):
    """
    Delta (sensitivity to underlying price) of Black-Scholes call option.
    
    Delta measures how much the option price changes when the underlying price changes.
    For a call option: delta = ∂E/∂V = Φ(d₁)
    
    Parameters:
    -----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns:
    --------
    float
        Delta of the call option (between 0 and 1)
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    delta = norm.cdf(d1)
    return delta


def black_scholes_vega(S, K, T, r, sigma):
    """
    Vega (sensitivity to volatility) of Black-Scholes call option.
    
    Parameters:
    -----------
    S : float
        Current asset price
    K : float
        Strike price
    T : float
        Time to maturity (in years)
    r : float
        Risk-free rate (annualized)
    sigma : float
        Volatility (annualized)
    
    Returns:
    --------
    float
        Vega of the call option
    """
    if T <= 0 or sigma <= 0 or S <= 0:
        return 0.0
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T)
    return vega

class MertonModel:
    """
    Baseline Merton structural credit model.
    
    Assumptions:
    - Firm asset value follows geometric Brownian motion
    - Default occurs only at maturity T if V_T < D
    - Equity is a European call option on firm assets
    """
    
    def __init__(self, T=1.0):
        """
        Initialize Merton model.
        
        Parameters:
        -----------
        T : float
            Time to maturity (years)
        """
        self.T = T
    
    def equity_value(self, V, D, r, sigma_V):
        """
        Calculate equity value as call option on assets.
        
        Parameters:
        -----------
        V : float
            Current asset value
        D : float
            Debt face value
        r : float
            Risk-free rate
        sigma_V : float
            Asset volatility
        
        Returns:
        --------
        float
            Equity value
        """

        return black_scholes_call(V, D, self.T, r, sigma_V)
    
    def equity_volatility(self, V, D, r, sigma_V, E):
        """
        Calculate equity volatility from asset volatility.
        
        Parameters:
        -----------
        V : float
            Current asset value
        D : float
            Debt face value
        r : float
            Risk-free rate
        sigma_V : float
            Asset volatility
        E : float
            Equity value
        
        Returns:
        --------
        float
            Equity volatility
        """
        if E <= 0:
            return 0.0
        
        delta = black_scholes_delta(V, D, self.T, r, sigma_V)
        
        sigma_E = (delta * sigma_V * V) / E
        return sigma_E

