"""
Example: How to align and prepare data for the Merton model.

This shows the key data preparation steps.
"""

import pandas as pd
import numpy as np


def align_data(data_dir='data/synthetic'):
    """
    Load and align all data files for modeling.
    
    Key challenge: Debt is quarterly, but equity is daily.
    Solution: Forward-fill quarterly debt values to daily frequency.
    """
    # Load all data
    equity_prices = pd.read_csv(f'{data_dir}/equity_prices.csv', parse_dates=['date'])
    equity_vol = pd.read_csv(f'{data_dir}/equity_vol.csv', parse_dates=['date'])
    debt = pd.read_csv(f'{data_dir}/debt_quarterly.csv', parse_dates=['date'])
    risk_free = pd.read_csv(f'{data_dir}/risk_free.csv', parse_dates=['date'])
    
    print("Original data shapes:")
    print(f"  Equity prices: {equity_prices.shape}")
    print(f"  Equity vol:    {equity_vol.shape}")
    print(f"  Debt (qtrly): {debt.shape}")
    print(f"  Risk-free:     {risk_free.shape}")
    
    # Align debt to daily frequency (forward-fill quarterly values)
    # Strategy: For each firm, create daily debt series
    all_dates = sorted(equity_prices['date'].unique())
    firms = equity_prices['firm_id'].unique()
    
    debt_daily_list = []
    for firm_id in firms:
        # Get quarterly debt for this firm
        firm_debt_qtrly = debt[debt['firm_id'] == firm_id].set_index('date')
        
        # Create daily index for this firm's date range
        firm_dates = equity_prices[equity_prices['firm_id'] == firm_id]['date'].unique()
        firm_dates = pd.DatetimeIndex(sorted(firm_dates))
        
        # Reindex to daily, forward-filling quarterly values
        firm_debt_daily = firm_debt_qtrly.reindex(firm_dates, method='ffill')
        
        # Convert back to DataFrame
        firm_debt_daily = firm_debt_daily.reset_index()
        firm_debt_daily['firm_id'] = firm_id
        firm_debt_daily.columns = ['date', 'debt', 'firm_id']
        firm_debt_daily = firm_debt_daily[['date', 'firm_id', 'debt']]
        
        debt_daily_list.append(firm_debt_daily)
    
    debt_daily = pd.concat(debt_daily_list, ignore_index=True)
    
    print(f"\nAligned debt (daily): {debt_daily.shape}")
    print(f"\nSample of aligned data:")
    print(debt_daily.head(10))
    
    return equity_prices, equity_vol, debt_daily, risk_free


def get_firm_data_for_date(equity_prices, equity_vol, debt_daily, risk_free, 
                           firm_id, date):
    """
    Extract all required inputs for a specific firm and date.
    
    Returns:
    --------
    dict with keys: E, sigma_E, D, r
    """
    # Get equity price
    E = equity_prices[
        (equity_prices['firm_id'] == firm_id) & 
        (equity_prices['date'] == date)
    ]['equity_price'].values
    
    # Get equity volatility
    sigma_E = equity_vol[
        (equity_vol['firm_id'] == firm_id) & 
        (equity_vol['date'] == date)
    ]['equity_vol'].values
    
    # Get debt (aligned to daily)
    D = debt_daily[
        (debt_daily['firm_id'] == firm_id) & 
        (debt_daily['date'] == date)
    ]['debt'].values
    
    # Get risk-free rate
    r = risk_free[risk_free['date'] == date]['risk_free_rate'].values
    
    # Check if all values exist
    if len(E) == 0 or len(sigma_E) == 0 or len(D) == 0 or len(r) == 0:
        return None
    
    return {
        'E': E[0],
        'sigma_E': sigma_E[0],
        'D': D[0],
        'r': r[0]
    }


def example_usage():
    """Example of how to use the aligned data."""
    equity_prices, equity_vol, debt_daily, risk_free = align_data()
    
    # Example: Get data for one firm on one date
    firm_id = equity_prices['firm_id'].unique()[0]
    date = equity_prices[equity_prices['firm_id'] == firm_id]['date'].iloc[0]
    
    inputs = get_firm_data_for_date(
        equity_prices, equity_vol, debt_daily, risk_free, 
        firm_id, date
    )
    
    if inputs:
        print(f"\nExample inputs for {firm_id} on {date.date()}:")
        print(f"  Equity value (E):      ${inputs['E']:.2f}")
        print(f"  Equity volatility:     {inputs['sigma_E']:.2%}")
        print(f"  Debt (D):              ${inputs['D']:.2f}M")
        print(f"  Risk-free rate (r):    {inputs['r']:.2%}")
        print(f"\nReady for calibration with T=1.0 year")


if __name__ == "__main__":
    example_usage()

