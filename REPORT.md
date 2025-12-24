## `model.py`
### `black_scholes_call(S, K, T, r, sigma)`

- **What it does**: Computes the **Black–Scholes price** of a European call option.
- **How it works**:
  - If $T \le 0$, $\sigma \le 0$, or $S \le 0$, it falls back to intrinsic value:  
    $\max(S - K e^{-rT}, 0)$ (discounted strike, still using $T$ in the discount).
  - Otherwise it computes:
    - $d_1 = \frac{\ln(S/K) + (r + 0.5\sigma^2)T}{\sigma \sqrt{T}}$
    - $d_2 = d_1 - \sigma\sqrt{T}$
    - $C = S \Phi(d_1) - K e^{-rT} \Phi(d_2)$
- **Inputs**:
  - `S`: Current underlying asset price.
  - `K`: Strike price.
  - `T`: Time to maturity in years.
  - `r`: Annual risk-free rate (continuously compounded).
  - `sigma`: Annual volatility of the underlying.
- **Output**:
  - `C`: Theoretical call option price.

---

### `black_scholes_delta(S, K, T, r, sigma)`

- **What it does**: Computes **delta**, i.e. the sensitivity of the call price to the underlying asset price, $\partial C / \partial S$.
- **How it works**:
  - If $T \le 0$, $\sigma \le 0$, or $S \le 0$, returns a simple in/out-of-the-money proxy: `1.0` if $S > K$ else `0.0`.
  - Otherwise:
    - $d_1$ as in `black_scholes_call`.
    - $\Delta = \Phi(d_1)$, where $\Phi$ is the standard normal CDF.
- **Inputs**: Same as `black_scholes_call`.
- **Output**:
  - $\Delta$: Delta of the call option (between 0 and 1).

---

### `black_scholes_vega(S, K, T, r, sigma)`

- **What it does**: Computes **vega**, i.e. the sensitivity of the call price to volatility, $\partial C / \partial \sigma$.
- **How it works**:
  - If $T \le 0$, $\sigma \le 0$, or $S \le 0$, returns 0.0 (no meaningful volatility exposure).
  - Otherwise:
    - $d_1$ as above.
    - $\nu$ $= S \phi(d_1) \sqrt{T}$
- **Inputs**: Same as `black_scholes_call`.
- **Output**:
  - `v`: Vega of the call.

---

### `class MertonModel`

Represents the **baseline Merton structural credit model**, where equity is modeled as a call option on firm assets.

#### `__init__(self, T=1.0)`

- **What it does**: Stores the **time to maturity**
- **Inputs**:
  - `T`: time to debt maturity in years (default 1.0).
- **Effect**:
  - Sets `self.T = T`.

---

#### `equity_value(self, V, D, r, sigma_V)`

- **What it does**: Computes the **equity value** given asset value and volatility, by treating equity as a call on assets with strike equal to debt.
- **How it works**:
  - $E = C$ with: 
    - `V` is underlying asset price.
    - `D` is strike price. 
- **Inputs**:
  - `V`: Asset value.
  - `D`: Debt face value.
  - `r`: Risk-free rate.
  - `sigma_V`: Asset volatility.
- **Output**:
  - `E`: model-implied equity value.

---

#### `equity_volatility(self, V, D, r, sigma_V, E)`

- **What it does**: Computes **equity volatility** implied by asset volatility and leverage, using the Merton relationship.
- **Key relationship**:
  - $\sigma_E E = \Delta \sigma_V V$, where:
    - $\sigma_E$: equity volatility,
    - $\Delta = \Phi(d_1)$ is the **equity delta** w.r.t. asset value.
  - So $\sigma_E = \dfrac{\Delta \sigma_V V}{E}$.
- **How it works**:
  - If `E <= 0`, returns `0.0` (no volatility defined for non-positive equity).
  - Computes $\Delta$ then applies the relationship:
    - `sigma_E = (delta * sigma_V * V) / E`.
- **Inputs**:
  - `V`: Asset value.
  - `D`: Debt face value.
  - `r`: Risk-free rate.
  - `sigma_V`: Asset volatility.
  - `E`: Equity value.
- **Output**:
  - `sigma_E`: Equity volatility. 

---

## `calibration.py`

### `calibrate_asset_parameters(E, sigma_E, D, T, r, V0=None, sigma_V0=None)`

- **What it does**: Calibrates **unobservable asset value (V)** and **asset volatility (sigma_V)** from observable equity data ($E$,  $\sigma_E$)  by solving a system of two equations.
- **How it works**:
  1. **Initial guesses**: If not provided, sets:
     - $V_0 = E + D$ (simple approximation: assets = equity + debt)
     - $\sigma_{V0} = \frac{\sigma_E \cdot E}{E + D}$ (leverage-adjusted volatility estimate)
  2. **System of equations**: Defines an inner function `equations(params)` to compute two equations:
     - **Equation 1**: $E = \text{BlackScholes}(V, D, T, r, \sigma_V)$
       - Computes the theoretical equity value using Black-Scholes.
       - Residual: `eq1 = E_calc - E` (should be zero at solution).
     - **Equation 2**: $\sigma_E \cdot E = \Delta \cdot \sigma_V \cdot V$
       - Computes option delta $\Delta$ (sensitivity of equity to asset value).
       - Computes theoretical equity volatility: `E_vol_calc = (delta * sigma_V * V) / E`.
       - Residual: `eq2 = E_vol_calc - sigma_E` (should be zero at solution).
  3. **Numerical solving**: Uses `scipy.optimize.fsolve` to find $V$ and $\sigma_V$ such that both residuals are zero.
  4. **Edge cases**:
     - Ensures $V \geq 10^{-6}$ and $\sigma_V \geq 10^{-6}$ (prevents negative values)
     - If calibration fails (exception), returns initial guesses as fallback
- **Inputs**:
  - `E`: Equity value.
  - `sigma_E`: Equity volatility.
  - `D`: Debt face value.
  - `T`: Time to maturity in years.
  - `r`: Risk-free rate.
  - `V0`: Initial guess for asset value (default: E + D).
  - `sigma_V0`: Initial guess for asset volatility.
- **Output**:
  - `(V, sigma_V)`: Estimated asset value and asset volatility.

---

## `risk_measures.py`

### `distance_to_default(V, D, T, r, sigma_V)`

- **What it does**: Calculates **Distance-to-Default (DD)**, a measure of how many standard deviations the expected asset value is above the default debt.
- **Interpretation**: Higher DD = lower default risk. DD measures the "safety margin" in units of asset volatility.
- **How it works**:
  - **Expected asset value at maturity**: $E[V_T] = V \cdot e^{rT}$
  - **Standard deviation of asset value at maturity**: 
    $\text{std}(V_T) = V \cdot e^{rT} \cdot \sqrt{e^{\sigma_V^2 T} - 1}$
  - **Distance-to-Default**: 
    $\text{DD} = \frac{E[V_T] - D}{\text{std}(V_T)}$
  - **Edge cases**:
    - If $T \leq 0$, returns `inf` if $V > D$ else `-inf`
    - If `std_VT <= 0`, returns `inf` if expected value > debt else `-inf`
- **Inputs**:
  - `V`: Current asset value.
  - `D`: Debt face value.
  - `T`: Time to maturity in years.
  - `r`: Risk-free rate.
  - `sigma_V`: Asset volatility.
- **Output**:
  - Distance-to-default


---

### `default_probability(V, D, T, r, sigma_V)`

- **What it does**: Calculates the **risk-neutral default probability (PD)**, i.e., the probability that asset value at maturity will be below the debt face value.
- **How it works**:
  - Under the Merton model, asset value at maturity $V_T$ is lognormally distributed
  - Default occurs if $V_T < D$
  - **Formula**: 
    $\text{PD} = \Phi(-d_2)$
    where:
    $d_2 = \frac{\ln(V/D) + (r - \sigma_V^2/2)T}{\sigma_V \sqrt{T}}$
  - **Edge cases**:
    - If $T \leq 0$, $\sigma_V \leq 0$, $V \leq 0$, or $D \leq 0$:
      - Returns `1.0` if $V < D$ (already in default), else `0.0`
    - Clamps result to $[0, 1]$ to ensure valid probability
- **Inputs**:
  - `V`: Asset value.
  - `D`: Debt face value.
  - `T`: Time to maturity in years.
  - `r`: Risk-free rate.
  - `sigma_V`: Asset volatility.
- **Output**:
  - Default probability between 0 and 1

---

### `compute_risk_measures(V, D, T, r, sigma_V)`

- **What it does**: Computes **Distance-to-Default** and **Default Probability**
- **How it works**:
  - Calls `distance_to_default()` and `default_probability()` 
- **Inputs**: Same as `distance_to_default` and `default_probability`
- **Output**:
  - Dictionary with keys:
    - `'DD'`: Distance-to-default
    - `'PD'`: Default probability

---

## `__main__.py`

### `main()`

- **What it does**: Main entry point that orchestrates the entire baseline Merton model pipeline: loads data, aligns it, calibrates parameters for each firm/date, computes risk measures, and saves results.
- **How it works**:
  1. **Data Loading**:
     - Loads four CSV files from `data/synthetic/` (or can be changed to `data/real/`):
       - `equity_prices.csv`: Daily equity prices per firm
       - `equity_vol.csv`: Daily equity volatility per firm
       - `debt_quarterly.csv`: Quarterly debt values per firm
       - `risk_free.csv`: Daily risk-free rates
     - Prints data shapes for verification
  2. **Data Alignment**:
    
  3. **Model Initialization**:
   
  4. **Calibration Loop**:
    
  5. **Output**:
     - Converts results list to pandas DataFrame
     - Creates `outputs/` directory if it doesn't exist
     - Saves results to `outputs/naive_results.csv` with columns:
       - `date`: Evaluation date
       - `firm_id`: Firm identifier
       - `V`: Estimated asset value
       - `sigma_V`: Estimated asset volatility
       - `DD`: Distance-to-default
       - `PD`: Default probability
     - Prints summary statistics (mean, std, min, max, etc.) for the numeric columns
- **Inputs**: None (reads from data files)
- **Output**: 
  - CSV file: `outputs/naive_results.csv`
  - Console output: Progress messages and summary statistics

---

## References

- **Merton model and calibration write-up**: [Merton Model Credit Risk Calculator](https://www.creditrisk.nathangs.ca/)