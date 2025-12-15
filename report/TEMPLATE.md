# Technical Report Template

## 1. Introduction

Brief overview of the assignment and your approach.

## 2. Model Formulation

### 2.1 Baseline Merton Model

- State the key assumptions
- Present the mathematical formulation
- Explain the calibration approach

### 2.2 Improved Model

- **What assumption did you modify?**
- **Why is this improvement justified?**
- Present the mathematical formulation of your improvement
- Explain how calibration changes (if at all)

## 3. Calibration Methodology

- How do you solve for V and σ_V?
- What numerical methods do you use?
- How do you handle edge cases or failures?

## 4. Empirical Setup

### 4.1 Implementation Details

- Time to maturity (T) assumption
- How you aligned quarterly debt with daily equity data
- Any other simplifying assumptions

## 5. Baseline Model Diagnosis

### 5.1 Identified Weaknesses

- What systematic failures did you observe?
- Provide evidence (plots, statistics, examples)

### 5.2 Examples

- Show specific cases where the baseline fails
- Include figures/tables

## 6. Improved Model Results

### 6.1 Quantitative Comparison

- Define your evaluation metric(s)
- Present comparison tables/figures
- Statistical summary

### 6.2 Why It's Better

- Explain why your improvement addresses the weakness
- Economic interpretation of results

## 7. Limitations

- What does your model still not capture?
- What assumptions remain?
- When might your improvement not work?

## 8. Conclusion

- Summary of findings
- Key takeaways
- Possible extensions

---

## Figures and Tables

Include:
- Time series plots of PD, DD for different firms
- Comparison plots (naive vs. improved)
- Summary statistics tables
- Cross-sectional comparisons

All figures should be clearly labeled and interpretable.

