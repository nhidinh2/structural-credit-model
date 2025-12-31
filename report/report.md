# Technical Report: Structural Credit Modeling

## 1. Introduction

This report presents an implementation and analysis of the Merton (1974) structural credit model, which treats a firm's equity as a European call option on its assets. The model is applied to real firm data representing five publicly traded companies (AAPL, JPM, TSLA, XOM, F) over a one-year period (2020), with the goal of calibrating unobservable asset values and volatilities from observable equity market data, and computing credit risk measures including distance-to-default and default probability.

After implementing and evaluating the baseline model, we identify systematic weaknesses in its behavior and propose potential improvements. The analysis focuses on understanding where the model fails and how these failures can be addressed through minimal, well-justified modifications to the core assumptions.

## 2. Model Formulation

### 2.1 Baseline Merton Model

#### Key Assumptions

1. **Capital structure simplification**: The firm is represented by a single zero-coupon debt claim with face value $D$ maturing at horizon $T$.

2. **Equity as a call option**: Equity holders receive $\max(V_T - D, 0)$, so equity is modeled as a European call option on firm assets with strike $D$.

3. **Asset dynamics**: Firm asset value follows a geometric Brownian motion with constant volatility over $[0, T]$:
   $$dV_t = \mu V_t \, dt + \sigma_V V_t \, dW_t$$
   where $\mu$ is the drift rate, $\sigma_V$ is asset volatility, and $dW_t$ is a Wiener process.

4. **Frictionless markets**: No taxes, transaction costs, or funding frictions; continuous trading; borrowing/lending at the risk-free rate.

5. **Default only at maturity**: Default occurs only at $T$ if $V_T < D$ (no early default / liquidity default).

6. **(Often implicit in baseline)** Dividends are ignored, and parameters like $\sigma_V$ are treated as constant over the horizon $T$.

#### Mathematical Formulation

Under risk-neutral valuation, equity is priced by Black–Scholes as:
$$E = V \Phi(d_1) - D e^{-r T} \Phi(d_2),$$
where
$$d_1 = \frac{\ln(V/D) + (r + \frac{1}{2}\sigma_V^2)T}{\sigma_V \sqrt{T}}, \quad d_2 = d_1 - \sigma_V \sqrt{T}.$$

A second relationship comes from equity volatility (Itô + option delta):
$$\sigma_E E = \Phi(d_1) \sigma_V V \quad \Longleftrightarrow \quad \sigma_E = \frac{\Phi(d_1) \sigma_V V}{E}.$$

Once $(V, \sigma_V)$ are determined, baseline credit risk metrics are:

**Distance to default**:
$$\text{DD} = \frac{\ln(V/D) + (r - \frac{1}{2}\sigma_V^2)T}{\sigma_V \sqrt{T}}.$$

**Default probability**:
$$\text{PD} = P(V_T < D) = \Phi(-\text{DD}) = \Phi(-d_2) \quad \text{(under the model)}.$$

#### Calibration Approach

For each firm-date, the model treats observed equity market value $E$, equity volatility $\sigma_E$, debt face value $D$, risk-free rate $r$, and horizon $T$ as inputs, and solves for the latent asset value $V$ and asset volatility $\sigma_V$ using the two equations:
$$\begin{cases}
E - (V \Phi(d_1) - D e^{-r T} \Phi(d_2)) = 0, \\
\sigma_E E - \Phi(d_1) \sigma_V V = 0.
\end{cases}$$

In the baseline implementation, $(V, \sigma_V)$ are obtained via unconstrained numerical root-finding (e.g., `fsolve`) independently each day (often warm-started from the previous solution). The calibrated $(V, \sigma_V)$ are then plugged into the formulas above to produce time series of $\text{DD}$ and $\text{PD}$.

### 2.2 Improved Model

The improved model applies EWMA (Exponential Weighted Moving Average) smoothing to $\sigma_E$:
   - **EWMA Smoothing**: Apply exponential weighted moving average to the variance, then take the square root:
     $$\text{var}_t^{\text{smooth}} = \lambda \cdot \text{var}_{t-1}^{\text{smooth}} + (1-\lambda) \cdot \sigma_{E,t}^2$$
     $$\sigma_{E,t}^{\text{smooth}} = \sqrt{\text{var}_t^{\text{smooth}}}$$
     where $\lambda = 0.94$ is the smoothing parameter.

**Justification**: The improved model differs from the baseline model **only** in the application of volatility smoothing. Both models use the same calibration methodology (`fsolve` root-finding) and initialization strategy (warm-start using previous period's solution). The volatility smoothing directly addresses the identified weakness by reducing measurement noise in inputs, mitigating—but not eliminating—structural sensitivity that drives PD instability.

## 3. Calibration Methodology

The calibration process solves for unobservable asset value $V$ and asset volatility $\sigma_V$ from observable equity value $E$ and equity volatility $\sigma_E$ using a system of two nonlinear equations. 

### 3.1 Numerical Method

We use `scipy.optimize.fsolve` to solve the system of equations:

$$\begin{cases}
E - \text{BlackScholes}(V, D, T, r, \sigma_V) = 0 \\
\sigma_E - \frac{\Delta \sigma_V V}{E} = 0
\end{cases}$$

where $\Delta = \Phi(d_1)$ is the option delta. For the improved model, $\sigma_E$ in the second equation is replaced with $\sigma_E^{\text{smooth}}$.

### 3.2 Initial Guesses and Warm-Start Strategy

The calibration uses a warm-start approach that reflects the persistence of firm value and asset volatility:

- **Initial observation**: For the first observation for each firm, use standard approximations:
  - **Asset Value**: $V_0 = E + D$ (assuming assets equal equity plus debt)
  - **Asset Volatility**: $\sigma_{V0} = \frac{\sigma_E \cdot E}{E + D}$ (leverage-adjusted volatility estimate), where $\sigma_E$ is either raw (baseline) or smoothed (improved)

- **Subsequent observations**: For each subsequent date, use the solution from the previous period as the initial guess:
  - $V_0 = V_{t-1}$ (previous period's asset value)
  - $\sigma_{V0} = \sigma_{V,t-1}$ (previous period's asset volatility)

This warm-start approach improves convergence by leveraging the temporal persistence of firm fundamentals, as asset values and volatilities typically change smoothly over time rather than jumping discontinuously.

### 3.3 Edge Case Handling

1. **Non-negative constraints**: Both $V$ and $\sigma_V$ are constrained to be at least $10^{-6}$ to prevent negative or zero values.

2. **Calibration failures**: If the numerical solver fails to converge or raises an exception, the function returns the standard approximation ($V = E + D$, $\sigma_V = \frac{\sigma_E \cdot E}{E + D}$).

3. **Solver parameters**: 
   - Tolerance: `xtol=1e-6` (solution tolerance)
   - Maximum iterations: `maxfev=5000` (maximum function evaluations)

### 3.4 Risk Measures Computation

Once $V$ and $\sigma_V$ are calibrated, risk measures are computed:

1. **Distance-to-Default (DD)**:
   $$\text{DD} = \frac{\ln(V/D) + (r - \sigma_V^2/2)T}{\sigma_V \sqrt{T}}$$
   This is the standardized distance in the log-normal distribution, equivalent to $d_2$ in the Black-Scholes framework.
   
   **Edge cases**:
   - If $T \leq 0$: returns `nan` (invalid input)
   - Inputs $V$, $D$, and $\sigma_V$ are validated upstream in the calibration and data loading pipeline, so no additional checks are performed here

2. **Default Probability (PD)**:
   $$\text{PD} = \Phi(-d_2) = \Phi\left(-\frac{\ln(V/D) + (r - \sigma_V^2/2)T}{\sigma_V \sqrt{T}}\right)$$
   where $d_2$ is the same as in the DD formula above.
   
   **Edge cases**:
   - If $T \leq 0$: returns `nan` (invalid input)
   - Inputs $V$, $D$, and $\sigma_V$ are validated upstream in the calibration and data loading pipeline, so no additional checks are performed here
   - Result is clamped to $[0, 1]$ to ensure valid probability

## 4. Empirical Setup

### 4.1 Implementation Details

#### Time to Maturity (T) Assumption

The model assumes a constant time to maturity of **T = 1.0 year** for all firms and all evaluation dates. Since the model uses a single debt proxy $D$ that collapses each firm's entire capital structure into one number, the true debt maturity structure is unknown. Picking a fixed $T=1$ year thus represents a reporting horizon—we compute default risk over a fixed one-year period rather than modeling the actual debt schedule. This approach is coherent and aligns with the common industry and academic convention where one-year PD is the standard measure for credit risk reporting, making results comparable to typical "1-year default probability" metrics.

#### Data Alignment: Quarterly Debt to Daily Equity

Debt data is available at quarterly frequency, while equity data is available daily. To align debt to daily frequency, the quarterly debt values are reindexed using `pandas.reindex()` with `bfill().ffill()`, which first backward-fills from later dates and then forward-fills to ensure all dates have debt values.

#### Equity Market Cap Calculation

Equity prices are per-share while debt is in total millions. To ensure consistent units, we convert equity to total market cap by multiplying per-share price by shares outstanding: $E_{\text{total}} = E_{\text{per-share}} \times N \times 1000$ (in millions). Calibration then uses total equity and total debt (both in millions). Shares outstanding: AAPL (16.93B), JPM (3.09B), TSLA (3.325B), XOM (4.25B), F (3.97B). 

#### Other Simplifying Assumptions

1. **Single Debt Proxy**: The model represents each firm's capital structure using a single debt face value, rather than modeling the full debt maturity structure. This simplification is standard in basic Merton model implementations.

2. **Constant Risk-Free Rate**: While daily risk-free rates are used, the model assumes that the risk-free rate is constant over the one-year horizon T for each calibration. This is consistent with the Merton model's assumption of constant interest rates.

3. **No Dividends**: The model does not account for dividend payments, which can affect equity values and the relationship between equity and asset values.

4. **European Option Assumption**: Equity is modeled as a European call option, meaning default can only occur at maturity T, not before. This limitation is retained in both baseline and improved models.

5. **Geometric Brownian Motion**: Asset values are assumed to follow a geometric Brownian motion with constant drift (risk-free rate) and constant volatility, which may not capture all real-world dynamics.

6. **Perfect Market Assumptions**: The model assumes frictionless markets, no transaction costs, and that all market participants have the same information.

These assumptions are necessary for analytical tractability but represent limitations that should be considered when interpreting results.

## 5. Baseline Model Diagnosis

### 5.1 Identified Weaknesses

#### (1) Unstable PD values

The estimated default probabilities exhibit large, discontinuous jumps over time that are inconsistent with smooth changes in underlying inputs.

**Evidence.** Figure 5.1 plots $\log(\text{PD})$ over time for each firm. All firms exhibit abrupt regime shifts and flatlining behavior. The magnitude of instability is quantified by the maximum daily change in $\log(\text{PD})$:

- AAPL: $\max |\Delta\log(\text{PD})| = 23.90$
- F: $4.60$
- JPM: $28.50$
- TSLA: $9.94$
- XOM: $31.55$

![Figure 5.1: log(PD) time-series](outputs/diagnosis_timeseries.png)

#### (2) Asset-value plausibility

The inferred asset values themselves are not the primary source of failure. We do not observe widespread implausible or invalid asset values ($V < 0$)  in this sample. 

**Evidence.** Figure 5.2 shows histograms of the asset-to-equity ratio ($V/E$). For AAPL, TSLA, XOM, and JPM, $V/E$ remains within plausible ranges (typically below 3), consistent with the Merton framework where $V \approx E + D$. Ford (F) exhibits higher $V/E$ ratios, but the calibration also yields very small asset volatility. This pattern is indicative of degeneracy. 

![Figure 5.2: V/E histograms](outputs/diagnosis_v_e_ratio.png)


#### (3) Risk ranking consistency

The model's ability to rank firms by distress is imperfect and occasionally unreliable.

**Evidence.** While the median daily Spearman rank correlation is $\rho_t = 0.700$, indicating moderate average alignment, failures occur:

- 17.1% of days exhibit the wrong sign between PD and leverage.
- PD ranks change abruptly even when leverage ranks remain constant.

![Figure 5.3: PD rank vs leverage rank over time](outputs/diagnosis_ranking.png)

#### (4) Excessive sensitivity to inputs

The dominant weakness of the baseline model is extreme sensitivity to equity volatility.

**Evidence.** We report median absolute elasticities of $\log(\text{PD})$ with respect to model inputs. Equity volatility $\sigma_E$ is the most influential parameter by a wide margin:

- $\text{median } |\text{sens}(\sigma_E)| = 9.211$
- $95^{\text{th}}$ percentile $|\text{sens}(\sigma_E)| = 90.445$
- Extreme cases exceed 150 (e.g., AAPL in June 2020)

In contrast, sensitivities to equity value ($\text{median } |\text{sens}| = 2.922$), debt ($\text{median } |\text{sens}| = 2.987$), interest rate ($\text{median } |\text{sens}| = 2.777$), and maturity ($\text{median } |\text{sens}| = 6.843$) are much smaller.

**Interpretation.** Small changes or estimation noise in $\sigma_E$ can induce large swings in PD, directly explaining the observed instability and ranking inconsistencies.

![Figure 5.4: Sensitivity bar chart](outputs/diagnosis_sensitivity.png)


### 5.2 Illustrative Examples

#### Example 1: PD instability driven by volatility sensitivity (AAPL)

In early June 2020, AAPL's $\log(\text{PD})$ exhibits sharp jumps exceeding 20 log-units despite relatively smooth equity dynamics. This period coincides with extreme $\sigma_E$ sensitivity ($|\text{sens}| > 150$), illustrating how volatility amplification drives PD instability.

**Plots:** $\log(\text{PD})$ time-series (Figure 5.1), sensitivity extremes (Figure 5.4)

#### Example 2: Ranking instability under stable leverage (TSLA)

In the Merton calibration, $$\sigma_E \approx \Phi(d_1) \sigma_V \frac{V}{E}.$$ With low leverage, equity is deep in-the-money so $\Phi(d_1) \approx 1$, and since $V/E \approx 1$, we get TSLA's high inferred asset volatility.

When $\sigma_V$ is large, PD becomes highly sensitive to day-to-day noise in $\sigma_E$. With strong volatility sensitivity (median $|\text{sens}(\sigma_E)| = 9.211$), small fluctuations in estimated equity volatility translate into large PD swings, breaking the link between PD rank and leverage-based distress.

![Figure 5.5: Asset volatility time-series](outputs/naive_sigma_V.png)

**Plot:** PD rank vs leverage rank (Figure 5.3)

#### Example 3: Cross-firm instability despite plausible asset values (XOM, JPM)

XOM and JPM exhibit large PD jumps ($\max |\Delta\log(\text{PD})| > 28$) even though their $V/E$ ratios remain within plausible ranges. This demonstrates that PD instability is not caused by implausible asset levels but by sensitivity amplification.

**Plots:** $\log(\text{PD})$ time-series (Figure 5.1), $V/E$ histograms (Figure 5.2)

### Summary

The baseline model’s main weakness is its excessive sensitivity (especially to equity volatility) produces unstable PD dynamics and occasional firm-specific calibration pathologies (most notably Ford), which can trigger episodic breakdowns in risk ranking and reduce reliability for credit risk assessment.

## 6. Improved Model Results

### 6.1 Quantitative Comparison

The improved model addresses the primary weakness identified in the baseline model (excessive sensitivity to noisy equity volatility) through volatility smoothing. We evaluate improvements using the same diagnostic framework applied to the baseline model.

#### (1) PD Stability

**Improvement**: EWMA smoothing reduces volatility measurement noise, mitigating—but not eliminating—structural sensitivity, which reduces PD instability.

**Evidence**: The maximum daily change in $\log(\text{PD})$ for all firms:

| Firm | Baseline max \|Δlog(PD)\| | Improved max \|Δlog(PD)\| |
|------|----------------|----------------------|
| AAPL | 23.90 | 2.311 |
| F | 4.60 | 0.2854 |
| JPM | 28.50 | 1.087 |
| TSLA | 9.94 | 1.757 |
| XOM | 31.55 | 1.077 |

![Figure 6.1: Improved log(PD) time-series](outputs/diagnosis_timeseries_improved.png)

The improved model exhibits smoother PD trajectories with fewer abrupt jumps compared to the baseline.

#### (2) Asset Value Plausibility

**Maintained**: The improved model maintains plausible asset values, similar to the baseline model.

**Evidence**: V/E ratios remain within economically reasonable ranges, similar to the baseline model.

![Figure 6.2: Improved V/E histograms](outputs/diagnosis_v_e_ratio_improved.png)

#### (3) Risk Ranking Consistency

**Improvement**: The improved model shows better alignment between PD ranks and leverage ranks.

**Evidence**: 
- Percentage of days with wrong sign: Improved: 1.2%, Baseline: 17.1%
- Top-1 distress in top-2 PD failure rate: Improved: 0.4%, Baseline: 0.8%

Both models show strong top-k containment, with the improved model showing better alignment between PD ranks and leverage-based distress ranks.

![Figure 6.3: Improved PD rank vs leverage rank](outputs/diagnosis_ranking_improved.png)

#### (4) Sensitivity to Inputs

**Improvement**: EWMA smoothing reduces volatility measurement noise, mitigating—but not eliminating—structural sensitivity to equity volatility.

**Evidence**: The median absolute elasticity of $\log(\text{PD})$ with respect to model inputs:

| Parameter | Baseline median $\lvert\text{sens}\rvert$ | Baseline p95 $\lvert\text{sens}\rvert$ | Improved median $\lvert\text{sens}\rvert$ | Improved p95 $\lvert\text{sens}\rvert$ |
|-----------|--------------------------|----------------------|-------------------------|----------------------|
| $\sigma_E$ | 9.211 | 90.445 | 6.790 | 58.394 |
| $E$ | 2.922 | 22.781 | 2.893 | 15.917 |
| $D$ | 2.987 | 24.606 | 2.874 | 15.817 |
| $r$ | 2.777 | 31.993 | 2.747 | 18.527 |
| $T$ | 6.843 | 45.529 | 6.002 | 29.182 |

![Figure 6.4: Improved sensitivity bar chart](outputs/diagnosis_sensitivity_improved.png)

The improved model shows reduced sensitivity to equity volatility, with more balanced sensitivity across all parameters.


### 6.2 Why It's Better

EWMA smoothing improves the model because the baseline Merton calibration amplifies short-horizon volatility noise into large swings in implied asset volatility, distance-to-default, and ultimately PD. 

By smoothing $\sigma_E$, we reduce measurement noise in the key input driving instability, producing:
- more stable PD paths (large reductions in $\max|\Delta\log(\text{PD})|$),
- more consistent risk ordering over time (wrong-sign days drop sharply from 17.1% to 1.2%), and
- lower effective sensitivity to $\sigma_E$ (median sensitivity drops from 9.211 to 6.790) while leaving other inputs largely unchanged.

**Economic Interpretation**: This approach can be viewed as signal extraction: observed daily equity volatility contains substantial transitory noise, while credit risk should respond primarily to persistent changes in firm risk. Volatility smoothing filters out short-term noise, producing more stable credit risk measures that better reflect underlying firm fundamentals.

## 7. Limitations

### 7.1 What the Model Still Does Not Capture

The model cannot capture early default or liquidity/refinancing-driven distress because equity is modeled as a European option and default is only assessed at maturity $T$. Debt is represented by a single proxy $D$ (total book debt from annual Yahoo Finance balance sheets, forward-filled daily), which ignores maturity structure, covenants, seniority, and near-term funding pressure. Corporate actions (dividends, buybacks, issuance) are not modeled, though they affect equity and inferred asset values.

### 7.2 What Assumptions Remain

We retain core Merton assumptions: asset value follows GBM with constant drift/volatility over the horizon, markets are frictionless, and default occurs only at $T$. We fix $T=1$ as a one-year PD horizon, since $D$ is an annual balance-sheet proxy rather than a maturity-matched promised payment. EWMA smoothing changes only the $\sigma_E$ input and does not alter the calibration equations, so ill-conditioning can still occur for some firms/dates.

### 7.3 When the Improvement May Not Work

EWMA smoothing can lag sudden regime shifts (earnings shocks, crises), temporarily understating risk, and may dampen genuine distress-related volatility spikes. If instability mainly comes from calibration ill-conditioning (e.g., high leverage or mismatched debt proxy), smoothing may improve PD smoothness but not fully prevent implausible implied parameters.

## 8. Conclusion

This report evaluates a baseline implementation of the Merton (1974) structural credit risk model and identifies systematic weaknesses, including unstable default probability (PD) estimates, excessive sensitivity to noisy equity volatility, and inconsistent risk rankings over time. These issues arise primarily from the nonlinear amplification of short-horizon volatility measurement noise in the calibration process.

Applying EWMA smoothing to equity volatility significantly improves model behavior. The improved model exhibits smoother PD trajectories, more consistent risk rankings, and reduced effective sensitivity to equity volatility, while preserving time-varying risk dynamics and leaving the structural model unchanged.

### Key Takeaways

- A large portion of PD instability in the baseline model is driven by noise in equity volatility estimates rather than fundamental changes in firm risk.

- Input-level stabilization via volatility smoothing can substantially improve the empirical reliability of structural credit models.

- Despite these improvements, core structural assumptions remain, and calibration ill-conditioning can still occur for certain firms or periods.

### Possible Extensions

Future work could address remaining limitations by incorporating maturity-aware debt measures, constrained or regularized calibration techniques, forward-looking (implied) volatility estimates, or alternative structural frameworks that allow for early default and liquidity risk. Also, because the baseline calibration can match the equations for firms like Ford by inflating $V$ and driving $\sigma_V$ toward zero, future work should add simple guardrails (e.g., a minimum $\sigma_V$ and a cap on extreme $V/E$) to prevent unrealistic parameter estimates.

